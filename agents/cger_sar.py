import random
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from agents.sar_lts import SAR_LTS, ReplayBuffer


class CGER_SAR_LTS(SAR_LTS):
    """
    CGER with SAR-TS
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        self.buffer_size = 1800  # 300s, 900s, 3600s for seizures
        self.window_size = 60  # 60s

        self.prototype_pool = {}  # Dictionary to store class prototypes
        self.cls_prototype_count = {}

        self.buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            input_shape=(self.config.DatasetParams["channels"], 256),
            batch_size=self.batch_size,
            device=self.device,
            random_generator=self.random_generator
        )

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer,
                      loss_fn, fold, model_path=None):
        train_loader, val_loader = self._prepare_dataloader(train_eeg, train_masks, val_eeg, val_masks, fs)
        self._train_model(train_loader, val_loader, patient_session_key, optimizer, loss_fn, fold, model_path)

        # Update replay memory after training
        for inputs, labels, selected_flags in train_loader:
            for i in range(len(inputs)):
                if selected_flags[i]:  # 仅更新被标记的样本
                    self.buffer.update_buffer(inputs[i], labels[i])

        # Calculate class prototypes - only if we have a saved model
        if self.best_model_path is not None:
            self._update_prototype_pool(train_loader)

    def _train_one_epoch(self, train_loader, optimizer, loss_fn, scaler):
        """
        Train the model for one epoch with LwF knowledge distillation.
        """
        self.model.train()
        train_loss = 0.0
        all_preds = torch.empty(0, dtype=torch.long, device=self.device)
        all_targets = torch.empty(0, dtype=torch.long, device=self.device)

        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Add experience replay data
            replay_inputs, replay_labels = self._sample_from_memories(buffer=self.buffer, batch_size=self.batch_size)
            if replay_inputs:
                replay_inputs = torch.stack(replay_inputs).to(self.device)
                replay_labels = torch.tensor(replay_labels).to(self.device)
                inputs = torch.cat([inputs, replay_inputs])
                targets = torch.cat([targets, replay_labels])

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features, outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)

            loss += self._calculate_prototype_loss(features, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)

        accuracy = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())
        return train_loss / len(train_loader), accuracy

    def _update_prototype_pool(self, train_loader):
        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        self.model.eval()

        with torch.no_grad():
            for batch in train_loader:
                inputs, targets = batch[0], batch[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features, logits = self.model(inputs)

                for cls in torch.unique(targets):
                    cls_mask = (targets == cls)
                    cls_features = features[cls_mask]

                    P_c_new = cls_features.mean(dim=0)
                    N_c_new = cls_features.size(0)

                    if cls in self.prototype_pool:
                        P_c_prev_fea = self.prototype_pool[cls.item()]
                        N_c_prev = self.cls_prototype_count[cls.item()]

                        # 使用指数移动平均更新现有原型
                        self.prototype_pool[cls.item()] = (P_c_prev_fea * N_c_prev + P_c_new * N_c_new) \
                                                          / (N_c_prev + N_c_new)
                        self.cls_prototype_count[cls.item()] = N_c_prev + N_c_new
                    else:
                        self.prototype_pool[cls.item()] = P_c_new
                        self.cls_prototype_count[cls.item()] = N_c_new

    def _calculate_prototype_loss(self, features, labels):
        prototype_loss = 0
        for cls in torch.unique(labels):
            cls_features = features[labels == cls]

            cls_prototype = self.prototype_pool.get(cls)   # cls_features.mean(dim=0)
            if cls_prototype is not None:
                prototype_loss += torch.mean(1 - F.cosine_similarity(cls_features, cls_prototype))

        return prototype_loss



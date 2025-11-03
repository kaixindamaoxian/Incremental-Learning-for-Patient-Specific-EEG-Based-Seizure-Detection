import random
import torch
import torch.nn.functional as F
import os
import pickle
from Trainer import Trainer
from sklearn.metrics import accuracy_score


class CGER(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.buffer = []
        self.buffer_size = 1800 # 300s, 900s, 1800s for seizures
        self.batch_size = config.ModelParams["batch_size"]

        self.prototype_pool = {}  # Dictionary to store class prototypes
        self.cls_prototype_count = {}

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer, loss_fn, fold, model_path=None):
        train_loader, val_loader = self._prepare_dataloader(train_eeg, train_masks, val_eeg, val_masks, fs)
        self._train_model(train_loader, val_loader, patient_session_key, optimizer, loss_fn, fold, model_path)

        # Update replay memory after training
        for inputs, labels in train_loader:
            try:
                self._update_memory(inputs, labels)
            except Exception as e:
                print(f"Error updating memory: {e}")

        # Calculate class prototypes
        self._update_prototype_pool(train_loader)

    def learner_predict(self, test_eeg, test_masks, fs=None, best_init_model=None):
        return super().learner_sz_predict(test_eeg, test_masks, fs)

    def _train_one_epoch(self, train_loader, optimizer, loss_fn, scaler):
        """
        Train the model for one epoch with LwF knowledge distillation.
        """
        self.model.train()
        train_loss = 0.0
        all_preds = torch.empty(0, dtype=torch.long, device=self.device)
        all_targets = torch.empty(0, dtype=torch.long, device=self.device)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Add experience replay data
            replay_inputs, replay_labels = self._sample_from_memories()
            if replay_inputs:
                try:
                    replay_inputs = torch.stack(replay_inputs).to(self.device)
                    replay_labels = torch.tensor(replay_labels).to(self.device)
                    inputs = torch.cat([inputs, replay_inputs])
                    targets = torch.cat([targets, replay_labels])
                except RuntimeError as e:
                    # Handle tensor stacking errors due to size mismatch
                    print(f"Warning: Could not stack replay samples due to size mismatch: {e}")
                    # Continue training without replay samples
                    pass

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

    def _update_memory(self, inputs, labels):
        """
        Update the replay memory with new data using reservoir sampling.
        """
        if self.buffer_size <= 0:
            raise ValueError("Memory size must be a positive integer.")

        for inp, lbl in zip(inputs, labels):
            inp, lbl = inp.to(self.device), lbl.to(self.device)
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((inp.cpu(), lbl.cpu()))
            else:
                idx = random.randint(0, len(self.buffer) - 1)
                if random.random() < self.buffer_size / (self.buffer_size + len(self.buffer)):
                    self.buffer[idx] = (inp.cpu(), lbl.cpu())

    def _sample_from_memories(self):
        """
        Sample a batch of data from replay memory.
        """
        if not self.buffer:
            return [], []

        sample_size = min(len(self.buffer), self.batch_size)
        try:
            replay_samples = random.sample(self.buffer, sample_size)
        except ValueError:
            return [], []

        replay_inputs = [inp for inp, _ in replay_samples]
        replay_labels = [lbl for _, lbl in replay_samples]
        return replay_inputs, replay_labels

    def _update_prototype_pool(self, train_loader):
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in train_loader:
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

    def save_buffer_to_file(self, file_path):
        """
        将缓冲区中的样本保存到文件中。
        :param file_path: 保存文件的路径
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.buffer, f)

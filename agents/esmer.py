import random
import torch
from Trainer import Trainer
from sklearn.metrics import accuracy_score


class ESMER(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.buffer = []
        self.buffer_size = 900 # 300s, 900s, 1800s for seizures
        self.batch_size = config.ModelParams["batch_size"]
        self.error_sensitivity = {}

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer, loss_fn, fold, model_path=None):
        train_loader, val_loader = self._prepare_dataloader(train_eeg, train_masks, val_eeg, val_masks, fs)
        self._train_model(train_loader, val_loader, patient_session_key, optimizer, loss_fn, fold, model_path)

        # After completing the training for the current task, update replay memory
        self._update_memory(train_loader)

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
            if self.buffer:
                replay_inputs, replay_labels = zip(
                    *random.sample(self.buffer, min(len(self.buffer), self.batch_size)))
                replay_inputs, replay_labels = torch.stack(replay_inputs).to(self.device), torch.stack(
                    replay_labels).to(self.device)
                inputs = torch.cat([inputs, replay_inputs])
                targets = torch.cat([targets, replay_labels])

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features, outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            self._update_error_sensitivity(inputs, targets, loss.item())

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)

        accuracy = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())

        return train_loss / len(train_loader), accuracy

    def _update_error_sensitivity(self, inputs, labels, loss):
        for input, label in zip(inputs, labels):
            key = (input.cpu().numpy().tobytes(), label.item())
            if key in self.error_sensitivity:
                self.error_sensitivity[key].append(loss)
            else:
                self.error_sensitivity[key] = [loss]

    def _update_memory(self, train_loader):
        all_data = []
        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            for i in range(len(inputs)):
                all_data.append((inputs[i], targets[i]))

        # sort by error sensitivity
        all_data.sort(key=lambda x: -self._get_error_sensitivity(x))

        for i, (inp, lab) in enumerate(all_data):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((inp, lab))
            else:
                # Reservoir sampling
                idx = random.randint(0, i)
                if idx < self.buffer_size:
                    self.buffer[idx] = (inp, lab)

    def _get_error_sensitivity(self, data):
        inputs, labels = data
        key = (inputs.cpu().numpy().tobytes(), labels.item())
        if key in self.error_sensitivity:
            return sum(self.error_sensitivity[key]) / len(self.error_sensitivity[key])
        return 0
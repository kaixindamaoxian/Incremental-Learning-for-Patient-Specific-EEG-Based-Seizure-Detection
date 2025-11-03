import random
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utils import OversampleProcessor
from sklearn.metrics import accuracy_score
from agents.sar_lts import SAR_LTS, ReplayBuffer


class ESMER_SAR_LTS(SAR_LTS):
    """
    Similarity-Aware Replay with Temporal Sampling (SAR-TS)
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        self.buffer_size = 1800  # 300s, 900s, 3600s for seizures
        self.window_size = 60  # 60s
        self.error_sensitivity = {}

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

        # After completing the training for the current task, update replay memory
        all_data = []
        for inputs, labels, selected_flags in train_loader:
            for i in range(len(inputs)):
                if selected_flags[i]:  # 仅更新被标记的样本
                    all_data.append((inputs[i], labels[i]))

        # sort by error sensitivity
        all_data.sort(key=lambda x: -self._get_error_sensitivity(x))

        for i, (inp, lab) in enumerate(all_data):
            self.buffer.update_buffer(inp, lab)

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

            replay_inputs, replay_labels = self._sample_from_memories(buffer=self.buffer, batch_size=self.batch_size)
            if replay_inputs:
                replay_inputs = torch.stack(replay_inputs).to(self.device)
                replay_labels = torch.tensor(replay_labels).to(self.device)
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

    def _get_error_sensitivity(self, data):
        inputs, labels = data
        key = (inputs.cpu().numpy().tobytes(), labels.item())
        if key in self.error_sensitivity:
            return sum(self.error_sensitivity[key]) / len(self.error_sensitivity[key])
        return 0

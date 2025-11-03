import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from agents.sar_lts import SAR_LTS, ReplayBuffer


class CLSER_SAR_LTS(SAR_LTS):
    """
    Similarity-Aware Replay with Temporal Sampling (SAR-TS)
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        self.fast_memory_size = 300
        self.slow_memory_size = 1800  # 300s, 900s, 1800s for seizure
        self.window_size = 60  # 60s

        # 校验 memory_size 和 batch_size
        if self.fast_memory_size <= 0 or self.slow_memory_size <= 0:
            raise ValueError("Memory size must be greater than 0.")
        if self.batch_size < 2:
            raise ValueError("Batch size must be at least 2 to support memory replay.")

        self.buffer_fast = ReplayBuffer(
            buffer_size=self.fast_memory_size,
            input_shape=(self.config.DatasetParams["channels"], 256),
            batch_size=self.batch_size,
            device=self.device,
            random_generator=self.random_generator
        )
        self.buffer_slow = ReplayBuffer(
            buffer_size=self.slow_memory_size,
            input_shape=(self.config.DatasetParams["channels"], 256),
            batch_size=self.batch_size,
            device=self.device,
            random_generator=self.random_generator
        )

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer,
                      loss_fn, fold, model_path=None):
        train_loader, val_loader = self._prepare_dataloader(train_eeg, train_masks, val_eeg, val_masks, fs)
        self._train_model(train_loader, val_loader, patient_session_key, optimizer, loss_fn, fold, model_path)

        # Update fast and slow memories with current batch using reservoir sampling
        for inputs, labels, selected_flags in train_loader:
            for i in range(len(inputs)):
                if selected_flags[i]:  # 仅更新被标记的样本
                    self.buffer_fast.update_buffer(inputs[i], labels[i])
                    self.buffer_slow.update_buffer(inputs[i], labels[i])

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

            replay_inputs, replay_labels = self._sample_memory()
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

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)

        accuracy = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())
        return train_loss / len(train_loader), accuracy

    def _sample_memory(self,):
        """
        从快慢记忆库中采样数据。
        """
        if not self.buffer_fast.buffer:
            return [], []

        fast_inputs, fast_labels = super()._sample_from_memories(buffer=self.buffer_fast, batch_size=self.batch_size // 2)
        slow_inputs, slow_labels = super()._sample_from_memories(buffer=self.buffer_slow, batch_size=self.batch_size // 2)

        # 合并快慢记忆库的样本
        replay_inputs = fast_inputs + slow_inputs  # 合并 inputs
        replay_labels = fast_labels + slow_labels  # 合并 labels

        return replay_inputs, replay_labels
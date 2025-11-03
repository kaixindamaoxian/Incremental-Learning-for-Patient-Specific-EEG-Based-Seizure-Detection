import random
import torch
from Trainer import Trainer
from sklearn.metrics import accuracy_score


class CLSER(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.fast_memory = []
        self.slow_memory = []
        self.fast_memory_size = 300
        self.slow_memory_size = 300  # 300s, 900s, 3600s for seizure
        self.batch_size = config.ModelParams["batch_size"]

        # 校验 memory_size 和 batch_size
        if self.fast_memory_size <= 0 or self.slow_memory_size <= 0:
            raise ValueError("Memory size must be greater than 0.")
        if self.batch_size < 2:
            raise ValueError("Batch size must be at least 2 to support memory replay.")

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer, loss_fn, fold, model_path=None):
        train_loader, val_loader = self._prepare_dataloader(train_eeg, train_masks, val_eeg, val_masks, fs)
        self._train_model(train_loader, val_loader, patient_session_key, optimizer, loss_fn, fold, model_path)

        # Update fast and slow memories with current batch using reservoir sampling
        for inputs, labels in train_loader:
            self._update_memory(self.fast_memory, inputs, labels, self.fast_memory_size)
            self._update_memory(self.slow_memory, inputs, labels, self.slow_memory_size)

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

            # Experience Replay: Add memory samples to the training batch
            replay_inputs, replay_labels = self._sample_from_memories()
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

    def _sample_from_memories(self):
        """
        从快慢记忆库中采样数据。
        """
        fast_samples = self._sample_memory(self.fast_memory, min(len(self.fast_memory), self.batch_size // 2))
        slow_samples = self._sample_memory(self.slow_memory, min(len(self.slow_memory), self.batch_size // 2))

        replay_inputs = [inp for inp, _ in fast_samples + slow_samples]
        replay_labels = [lbl for _, lbl in fast_samples + slow_samples]

        return replay_inputs, replay_labels

    def _sample_memory(self, memory, num_samples):
        """
        从指定记忆库中采样指定数量的数据。
        """
        if not memory or num_samples <= 0:
            return []
        return random.sample(memory, num_samples)

    def _update_memory(self, memory, inputs, labels, memory_size):
        for inp, lbl in zip(inputs, labels):
            inp, lbl = inp.to(self.device), lbl.to(self.device)
            if len(memory) < memory_size:
                memory.append((inp.cpu(), lbl.cpu()))
            else:
                # Reservoir sampling
                idx = random.randint(0, len(memory) - 1)
                if random.random() < memory_size / (memory_size + len(memory)):
                    memory[idx] = (inp.cpu(), lbl.cpu())
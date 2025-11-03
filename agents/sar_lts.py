import random
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os
import pickle
from Trainer import Trainer
from utils import OversampleProcessor
from sklearn.metrics import accuracy_score


class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, batch_size, device, random_generator):
        """
        初始化回放缓冲区。
        :param buffer_size: 缓冲区大小
        :param input_shape: 输入样本的形状
        :param batch_size: 批量大小
        :param device: 设备（CPU/GPU）
        :param seed: 随机数生成器的种子
        """
        if buffer_size <= 0 or batch_size <= 0:
            raise ValueError("buffer_size 和 batch_size 必须为正整数")

        self.buffer_size = buffer_size
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.device = device
        self.buffer = []  # 存储回放样本

        # 创建随机数生成器并设置种子
        self.random_generator = random_generator

    def _flatten_samples(self, samples):
        """
        将样本列表中的每个样本进行扁平化处理。
        :param samples: 样本列表
        :return: 扁平化后的张量列表
        """
        if not samples:
            return torch.empty((0, self.input_shape[0] * self.input_shape[1]), device=self.device)

        if not all(isinstance(sample, (list, tuple, torch.Tensor)) for sample in samples):
            raise ValueError("samples 中的元素必须是 list、tuple 或 torch.Tensor 类型")

        flattened = [torch.flatten(sample.clone().detach()).to(self.device) for sample in samples]

        return torch.cat(flattened).view(len(samples), -1)

    def _sample_indices(self, sample_flattened, sampling_strategy):
        """
        根据采样策略选择缓冲区中的样本索引。
        """
        if len(self.buffer) < self.batch_size:
            return None

        if sampling_strategy == "random":
            sample_indices = self.random_generator.sample(range(len(self.buffer)), self.batch_size)
        elif sampling_strategy == "stride_random":
            stride = len(self.buffer) // self.batch_size
            if stride == 0:
                raise ValueError("batch_size 过大，无法进行等间距采样")

            # 选择每个等间距范围内的一个随机样本
            sample_indices = []
            for i in range(self.batch_size):
                start_idx = i * stride
                # 确保最后一段区间的 end_idx 与缓冲区大小一致
                end_idx = len(self.buffer) if i == self.batch_size - 1 else min(start_idx + stride, len(self.buffer))
                sample_indices.append(self.random_generator.choice(range(start_idx, end_idx)))
        else:
            raise ValueError(f"未知的采样策略: {sampling_strategy}")

        selected_samples = [self.buffer[i][0] for i in sample_indices]
        selected_samples_flattened = self._flatten_samples(selected_samples)

        similarities = F.cosine_similarity(sample_flattened.unsqueeze(0), selected_samples_flattened)
        max_sim_idx = torch.argmax(similarities)
        return sample_indices[max_sim_idx.item()]

    def update_buffer(self, sample, label, sampling_strategy = "stride_random"):
        """
        更新回放缓冲区，按照给定的标志存储或不存储样本。
        """
        if not isinstance(sample, (list, tuple, torch.Tensor)):
            raise ValueError("sample 必须是 list、tuple 或 torch.Tensor 类型")
        if not isinstance(label, (int, float, torch.Tensor)):
            raise ValueError("label 必须是 int、float 或 torch.Tensor 类型")

        sample_label_pair = (sample, label)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(sample_label_pair)
        else:
            sample_flattened = self._flatten_samples([sample])[0]
            max_sim_idx = self._sample_indices(sample_flattened, sampling_strategy=sampling_strategy)
            if max_sim_idx is not None:
                self.buffer[max_sim_idx] = sample_label_pair

    def save_buffer_to_file(self, file_path):
        """
        将缓冲区中的样本保存到文件中。
        :param file_path: 保存文件的路径
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.buffer, f)


class SAR_LTS(Trainer):
    """
    Similarity-Aware Replay with Local Temporal Sampling (SAR-LTS)
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        self.batch_size = config.ModelParams["batch_size"]

        self.memory_size = 1800  # 300s, 900s, 3600s for seizure, 120s, 360s, 600s for sleep
        self.window_size = 60  # 60s

        self.random_generator = random.Random(config.seed)

        self.buffer = ReplayBuffer(
            buffer_size=self.memory_size,
            input_shape=(self.config.DatasetParams["channels"], 256),
            batch_size=self.batch_size,
            device=self.device,
            random_generator=self.random_generator
        )

        self.global_np_random_state = np.random.RandomState(config.seed)

        # 创建保存路径
        os.makedirs("./buffer_samples", exist_ok=True)

    def _prepare_dataloader(self, train_eeg, train_masks, val_eeg, val_masks, fs):
        train_inputs, train_labels = self._extract_inputs_labels(train_eeg, train_masks, fs)
        processor = OversampleProcessor(win_len=self.config.win_len, fs=fs)
        oversample_inputs, oversample_labels = processor.oversample_seizures(train_eeg, train_masks, 0, 0,
                                                                             0.05)  # 0.05

        if oversample_inputs.size != 0:
            train_inputs = np.concatenate((train_inputs, oversample_inputs), axis=0)
            train_labels = np.concatenate((train_labels, oversample_labels), axis=0)

        val_inputs, val_labels = self._extract_inputs_labels(val_eeg, val_masks, fs)

        # 初始化选择标志数组
        selected_flags = np.zeros(len(train_inputs), dtype=bool)

        idx = 0
        while idx < len(train_inputs):
            window_end = min(idx + self.window_size, len(train_inputs))
            select_idx = (idx + (window_end - idx) // 2) if self.config.sampling_strategy == 'middle' else \
                self.global_np_random_state.choice(range(idx, window_end), size=1)[0]

            # 标记选择的样本
            selected_flags[select_idx] = True
            idx += (window_end - idx)

        # 为训练pipline创建需要shuffle的DataLoader
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_inputs), torch.LongTensor(train_labels),
                          torch.BoolTensor(selected_flags)),
            batch_size=self.config.ModelParams["batch_size"], shuffle=True)

        val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_inputs), torch.LongTensor(val_labels)),
                                batch_size=self.config.ModelParams["batch_size"], shuffle=False)
        return train_loader, val_loader

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer,
                      loss_fn, fold, model_path=None):
        train_loader, val_loader = self._prepare_dataloader(train_eeg, train_masks, val_eeg, val_masks, fs)
        self._train_model(train_loader, val_loader, patient_session_key, optimizer, loss_fn, fold, model_path)

        # After completing the training for the current task, update replay memory
        for inputs, labels, selected_flags in train_loader:
            for i in range(len(inputs)):
                if selected_flags[i]:  # 仅更新被标记的样本
                    self.buffer.update_buffer(inputs[i], labels[i])

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

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)

        accuracy = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())
        return train_loss / len(train_loader), accuracy

    def learner_predict(self, test_eeg, test_masks, fs=None, best_init_model=None):
        return super().learner_sz_predict(test_eeg, test_masks, fs)

    def _sample_from_memories(self, sampling_strategy="stride_random", buffer=None, batch_size=None):
        if not buffer.buffer:
            return [], []

        if sampling_strategy == "random":
            replay_samples = random.sample(buffer.buffer, min(len(buffer.buffer), batch_size))
        elif sampling_strategy == "stride_random":
            stride = len(buffer.buffer) // batch_size
            if stride == 0:
                replay_samples = random.sample(buffer.buffer, min(len(buffer.buffer), batch_size))
            else:
                sample_indices = []
                for i in range(batch_size):
                    start_idx = i * stride
                    # 确保最后一段区间的 end_idx 与缓冲区大小一致
                    end_idx = len(buffer.buffer) if i == batch_size - 1 else min(start_idx + stride,
                                                                                           len(buffer.buffer))
                    sample_indices.append(self.random_generator.choice(range(start_idx, end_idx)))

                replay_samples = [buffer.buffer[i] for i in sample_indices]
        else:
            raise ValueError(f"未知的采样策略: {sampling_strategy}")

        return [inp for inp, _ in replay_samples], [lbl for _, lbl in replay_samples]
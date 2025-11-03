import torch
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from Trainer import Trainer


class LWF(Trainer):
    def __init__(self, config, model, temperature=2):
        super().__init__(config, model)
        self.old_model = None
        self.temperature = temperature

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer, loss_fn, fold, model_path=None):
        super().learner_train(train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer, loss_fn, fold, model_path)

        try:
            # 加载最佳模型权重，并捕获可能的异常
            self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
            self.old_model = copy.deepcopy(self.model)
        except FileNotFoundError:
            raise RuntimeError(f"Model file not found at path: {self.best_model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

    def learner_predict(self, test_eeg, test_masks, fs=None):
        return super().learner_sz_predict(test_eeg, test_masks, fs)

    def _train_one_epoch(self, train_loader, optimizer, loss_fn, scaler):
        """
        Train the model for one epoch with LwF knowledge distillation.
        """
        self.model.train()

        # 检查 train_loader 是否为空
        if len(train_loader) == 0:
            raise ValueError("The train_loader is empty. No data available for training.")

        train_loss = 0.0
        all_preds = []  # 使用列表存储预测结果
        all_targets = []  # 使用列表存储目标值

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features, outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)

                # LWF: Distillation loss
                if self.old_model is not None:
                    self.old_model.eval()
                    with torch.no_grad():
                        old_features, old_outputs = self.old_model(inputs)

                    kd_loss = F.kl_div(
                        F.log_softmax(outputs / self.temperature, dim=1),
                        F.softmax(old_outputs / self.temperature, dim=1),
                        reduction='batchmean'
                    ) * (self.temperature * self.temperature)
                    loss += kd_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            all_preds.append(preds)  # 存储预测结果
            all_targets.append(targets)  # 存储目标值

        # 一次性拼接所有预测和目标值
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        accuracy = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())

        return train_loss / len(train_loader), accuracy

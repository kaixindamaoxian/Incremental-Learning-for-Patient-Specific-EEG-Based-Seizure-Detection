import torch
import os
from Trainer import Trainer


class Finetune(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer, loss_fn, fold, model_path=None):
        super().learner_train(train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer, loss_fn, fold, model_path)

    def learner_predict(self, test_eeg, test_masks, fs=None):
        return super().learner_sz_predict(test_eeg, test_masks, fs)

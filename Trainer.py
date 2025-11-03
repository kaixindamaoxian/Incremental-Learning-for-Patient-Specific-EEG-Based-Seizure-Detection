from sklearn.metrics import f1_score, accuracy_score
from utils import OversampleProcessor, CUDAPerfMonitor
from loss import *
from timescoring import annotations
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path


class Trainer:
    def __init__(self, config, model):
        """
        Initialize the Trainer class with model and configuration.

        Parameters:
        config (object): Configuration object containing model settings.
        model (torch.nn.Module): The model to be trained.
        """
        self.config = config
        self.device = config.device
        self.model = model.to(config.device)
        self.best_model_path = None
        self.early_stopping_patience = self.config.ModelParams["early_stopping_patience"]  # patience for early stopping

    def _extract_inputs_labels(self, eeg_data, masks, fs):
        """
        Extract features and labels from EEG data.

        Parameters:
        eeg_data (numpy.ndarray): EEG data of shape (n_channels, n_samples).
        masks (numpy.ndarray): Masks indicating seizure occurrences.
        fs (float): Sampling frequency.

        Returns:
        tuple: A tuple containing inputs and labels as numpy arrays.
        """
        win_size = int(self.config.win_len * fs)  # Window size in samples (1 second = 256 samples)
        step_size = int(self.config.win_step * fs)

        inputs, labels = [], []
        for i in range(0, eeg_data.shape[1] - win_size + 1, step_size):
            window_eeg = eeg_data[:, i:i + win_size]
            inputs.append(window_eeg)

            window_mask = masks[i:i + win_size]
            seizure_overlap = np.sum(window_mask) / win_size
            label = int(seizure_overlap >= 0.5)
            labels.append(label)

        # To predict at 1Hz, we need to repeat the first samples
        while len(inputs) < eeg_data.shape[1] // fs:
            inputs.insert(0, inputs[0]) # Repeat first sample
            labels.insert(0, labels[0])

        return np.array(inputs, dtype=np.float32), np.array(labels)

    def _prepare_dataloader(self, train_eeg, train_masks, val_eeg, val_masks, fs):
        """
        Prepare data loaders for training and validation.

        Parameters:
        train_eeg, train_masks, val_eeg, val_masks: EEG data and corresponding masks
        fs (float): Sampling frequency

        Returns:
        tuple: DataLoader for training and validation sets.
        """
        if self.config.dataset in ["CHB-MIT", "Siena"]:  #  for seizure dataset
            train_inputs, train_labels = self._extract_inputs_labels(train_eeg, train_masks, fs)
            processor = OversampleProcessor(win_len=self.config.win_len, fs=fs)
            oversample_inputs, oversample_labels = processor.oversample_seizures(train_eeg, train_masks, 0, 0,
                                                                                 0.05)  # 0.05

            if oversample_inputs.size != 0:
                train_inputs = np.concatenate((train_inputs, oversample_inputs), axis=0)
                train_labels = np.concatenate((train_labels, oversample_labels), axis=0)

            val_inputs, val_labels = self._extract_inputs_labels(val_eeg, val_masks, fs)
        else:  # for sleep dataset
            train_inputs, train_labels, val_inputs, val_labels = train_eeg, train_masks, val_eeg, val_masks

        train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_inputs), torch.LongTensor(train_labels)),
                                  batch_size=self.config.ModelParams["batch_size"], shuffle=True, ) # drop_last=True if self.config.dataset in ["CHB-MIT", "Siena"] else False
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_inputs), torch.LongTensor(val_labels)),
                                batch_size=self.config.ModelParams["batch_size"], shuffle=False)
        return train_loader, val_loader

    def _train_one_epoch(self, train_loader, optimizer, loss_fn, scaler):
        """
        Train the model for one epoch.

        Parameters:
        input_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        loss_fn (callable): Loss function for training.

        Returns:
        float: Average training loss for the epoch.
        """
        self.model.train()
        train_loss = 0.0
        all_preds = torch.empty(0, dtype=torch.long, device=self.device)
        all_targets = torch.empty(0, dtype=torch.long, device=self.device)

        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            inputs, targets = inputs.to(self.device), targets.to(self.device)

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

    def _validate_one_epoch(self, val_loader, loss_fn):
        """
        Validate the model on validation data.

        Parameters:
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (callable): Loss function for validation.

        Returns:
        tuple: Average validation loss and F1 score.
        """
        self.model.eval()
        val_loss = 0.0
        all_preds = torch.empty(0, dtype=torch.long, device=self.device)
        all_targets = torch.empty(0, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                features, outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=-1)
                all_preds = torch.cat((all_preds, preds), dim=0)
                all_targets = torch.cat((all_targets, targets), dim=0)

        accuracy = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())
        f1_weighted = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        num_cls = self.config.ModelParams.get("num_cls", 2)
        f1_binary = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(),
                             average='weighted' if num_cls > 2 else 'binary', zero_division=0.0)

        return val_loss / len(val_loader), accuracy, (f1_weighted + f1_binary) / 2

    def _save_best_model(self, patient_session_key, fold):
        """
        Save the best model based on validation performance.

        Parameters:
        patient_session_key (str): Identifier for the patient session.
        fold (int): Fold number for cross-validation.
        """
        Path(self.config.checkpoints).mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.config.checkpoints + f'/best_model_{patient_session_key}_{fold}.pth'

        torch.save(self.model.state_dict(), self.best_model_path)
        print(f"Best model saved with F1: {self.best_model_path}")

    def initial_training(self, initial_train_data, initial_train_labels, initial_val_data, initial_val_labels, patient_id, optimizer, loss_fn):
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(initial_train_data), torch.LongTensor(initial_train_labels)),
                                  batch_size=self.config.ModelParams["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(initial_val_data), torch.LongTensor(initial_val_labels)),
                                batch_size=self.config.ModelParams["batch_size"], shuffle=False)
        self._train_model(train_loader, val_loader, patient_id, fold='init', optimizer=optimizer, loss_fn=loss_fn)
        return self.best_model_path

    def learner_train(self, train_eeg, train_masks, val_eeg, val_masks, fs, patient_session_key, optimizer, loss_fn, fold, model_path=None):
        """
        Train the model using the provided EEG data.

        Parameters:
        train_eeg (numpy.ndarray): Training EEG data.
        train_masks (numpy.ndarray): Training masks for seizure occurrences.
        fs (float): Sampling frequency.
        val_eeg (numpy.ndarray): Validation EEG data.
        val_masks (numpy.ndarray): Validation masks.
        patient_session_key (str): Identifier for the patient session.
        fold (int): Fold number for cross-validation.
        """
        train_loader, val_loader = self._prepare_dataloader(train_eeg, train_masks, val_eeg, val_masks, fs)
        self._train_model(train_loader, val_loader, patient_session_key, optimizer, loss_fn, fold, model_path)

    def _train_model(self, train_loader, val_loader, patient_session_key, optimizer, loss_fn, fold, model_path=None):
        best_f1 = - 1e-9
        epochs_without_improvement = 0

        if self.config.mode == 'Joint_training' and fold != 'init':
            model_path_to_load = model_path if model_path else self.best_model_path
        else:
            model_path_to_load = self.best_model_path

        if model_path_to_load is not None:
            print(f"Load the best model path: {model_path_to_load}")
            self.model.load_state_dict(torch.load(model_path_to_load, weights_only=True))

        scaler = torch.amp.GradScaler("cuda", enabled=True)   # For gradient scaling
        with CUDAPerfMonitor("Model Inference"):
            for epoch in range(self.config.ModelParams["epochs"]):
                train_loss, train_acc = self._train_one_epoch(train_loader, optimizer, loss_fn, scaler)
                val_loss, val_acc, val_f1 = self._validate_one_epoch(val_loader, loss_fn)

                print(f"Epoch {epoch + 1}/{self.config.ModelParams['epochs']}, "
                      f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.3f}, "
                      f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}")

                # Early stopping based on validation F1 score
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    epochs_without_improvement = 0
                    self._save_best_model(patient_session_key, fold)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}. Best F1: {best_f1}")
                    break

    def learner_sz_predict(self, test_eeg, test_masks, fs):
        """
        Make seizure predictions using the trained model.

        Parameters:
        eeg_data (numpy.ndarray): EEG data for prediction.
        masks (numpy.ndarray): Masks for seizure occurrences.
        fs (float): Sampling frequency.

        Returns:
        annotations.Annotation: Predicted annotations.
        """
        inputs, _ = self._extract_inputs_labels(test_eeg, test_masks, fs)
        test_loader = DataLoader(TensorDataset(torch.Tensor(inputs).float()),
                                 batch_size=self.config.ModelParams["batch_size"])

        print(f"The best model path to make predictions: {self.best_model_path}")
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        self.model.eval()

        all_preds = torch.empty(0, dtype=torch.long, device=self.device)
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(self.device)
                features, outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=-1)
                all_preds = torch.cat((all_preds, preds), dim=0)

        # Map window-level predictions back to sample-level annotations
        predicted_samples = np.zeros(test_eeg.shape[1])
        step_size = int(self.config.win_step * fs)
        for i, pred in enumerate(all_preds.cpu().numpy()):
            start_idx = i * step_size
            end_idx = start_idx + step_size
            predicted_samples[start_idx:end_idx] = pred

        return annotations.Annotation(predicted_samples, fs).mask
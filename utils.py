import os
import pickle
import random
import pyedflib
import numpy as np
import enum

import torch
import yaml
from pathlib import Path
import epilepsy2bids.annotations
import timescoring.scoring
from epilepsy2bids.eeg import Eeg
from scipy import signal
from datetime import datetime


SAMPLING_FREQUENCY = 256  # Sampling frequency

def load_yaml_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            print(f"Successfully loaded configuration file: {file_path}")
            return config
    except Exception as e:
        print(f"Failed to load configuration file: {file_path}, Error:{e}")


def set_random_seed(seed):
    """Set the random seed for reproducibility"""
    random.seed(seed)   # for python random
    np.random.seed(seed)  # for Numpy
    torch.manual_seed(seed)  # for PyTorch
    torch.cuda.manual_seed_all(seed)  # for all GPUs
    torch.backends.cudnn.deterministic = True   # Enable deterministic mode
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode


def save_data(patient_id, eegs, labels, fs, save_path):
    """Save EEG data and labels to a .pkl file."""
    file_path = os.path.join(save_path, f'{patient_id}_eeg_label_data.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump({'eegs': eegs, 'labels': labels, 'fs': fs}, f)


def load_data(patient_id, load_path):
    """Load EEG data and labels from a .pkl file."""
    file_path = os.path.join(load_path, f'{patient_id}/{patient_id}_eeg_label_data.pkl')    # f'{patient_id}/{patient_id}_eeg_label_data.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data['eegs'], data['labels'], data['fs']
    return None, None, None


def to_mask(annotations):
    """Generate a mask based on annotations."""
    mask = np.zeros(int(annotations.events[0]["recordingDuration"] * SAMPLING_FREQUENCY))   # 2625 * 256 = 672000, nd(672000,)
    for event in annotations.events:
        if event["eventType"].value != "bckg":
            mask[
                round(event["onset"] * SAMPLING_FREQUENCY): round(event["onset"] + event["duration"])
                * SAMPLING_FREQUENCY
            ] = 1
    return mask


class Montage(str, enum.Enum):
    UNIPOLAR = "unipolar"
    BIPOLAR = "bipolar"

BIPOLAR_DBANANA = (
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
)
ELECTRODES_10_20 = (
    "Fp1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
)

def load_eeg_and_labels(edf_file, tsv_file, args):
    """Load EEG data and labels from EDF and TSV files."""
    if args.dataset == 'CHB-MIT':
        montage = Montage.BIPOLAR
        electrode = BIPOLAR_DBANANA
    elif args.dataset == 'Siena':
        montage = Montage.UNIPOLAR
        electrode = ELECTRODES_10_20

    eeg = Eeg.loadEdf(edf_file, montage, electrode)
    eeg.data = eeg.data.astype(np.float32)
    with pyedflib.EdfReader(edf_file) as edf:
        dateTime = edf.getStartdatetime()
        duration = edf.getFileDuration()

    ref = epilepsy2bids.annotations.Annotations.loadTsv(tsv_file)
    ref = timescoring.annotations.Annotation(to_mask(ref), eeg.fs)
    return eeg, ref, dateTime, duration


def merge_patient_session(patient_files, args):
    """Merge EEG data and labels by patient and session."""
    merged_data = {}

    for edf_file, tsv_file in patient_files:
        patient_id = Path(edf_file).parts[-4]
        session_id = Path(edf_file).parts[-3]

        patient_session_key = f"{patient_id}_{session_id}"

        if patient_session_key not in merged_data:
            merged_data[patient_session_key] = {'eeg': [], 'labels': [], 'dateTime': [], 'duration': []}

        eeg, labels, dateTime, duration = load_eeg_and_labels(edf_file, tsv_file, args)
        merged_data[patient_session_key]['eeg'].append(eeg)
        merged_data[patient_session_key]['labels'].append(labels)
        merged_data[patient_session_key]['dateTime'].append(dateTime)
        merged_data[patient_session_key]['duration'].append(duration)

    return merged_data


def merge_patient(patient_files, args):
    """Merge EEG data and labels by patient, combining all sessions for each patient."""
    merged_data = {}

    for edf_file, tsv_file in patient_files:
        patient_id = Path(edf_file).parts[-4]

        if patient_id not in merged_data:
            merged_data[patient_id] = {'eeg': [], 'labels': [], 'dateTime': [], 'duration': []}

        eeg, labels, dateTime, duration = load_eeg_and_labels(edf_file, tsv_file, args)
        merged_data[patient_id]['eeg'].append(eeg)
        merged_data[patient_id]['labels'].append(labels)
        merged_data[patient_id]['dateTime'].append(dateTime)
        merged_data[patient_id]['duration'].append(duration)

    return merged_data


def has_seizure_events(labels):
    """Check if the labels contains seizure start (1) and end (-1) events"""
    sz_starts = np.where(np.diff(np.array(labels, dtype=int)) == 1)[0]
    sz_ends = np.where(np.diff(np.array(labels, dtype=int)) == -1)[0]
    return len(sz_starts) > 0 and len(sz_ends) > 0


def split_sz_for_frozen_model(total_samples, test_step_samples, num_folds, args, labels):
    """Split data for frozen model mode, returning training, validation, and test sets"""
    val_idx, test_idx = None, None
    found_sz = False

    for fold in range(num_folds):
        train_end = total_samples - (test_step_samples * (num_folds - fold + 1))    # 3.24 - (0.5 * (3 - fold) + 1)
        train_idx = np.arange(0, train_end)

        # check if seizure events in the current training set
        found_sz = found_sz or has_seizure_events(labels[train_idx])

        if found_sz:
            val_idx = np.arange(train_end, train_end + test_step_samples)
            test_idx = np.arange(train_end + test_step_samples, total_samples)
            break

    return train_idx, val_idx, test_idx


def sz_time_series_split(total_samples, test_step_samples, num_folds, args, labels):
    """Time-series cross-validation with minimum train data and fixed test size."""
    found_sz = False

    for fold in range(num_folds):
        train_end = total_samples - (test_step_samples * (num_folds - fold + 1))
        train_start = max(0, train_end - test_step_samples if found_sz and args.mode in ['Incremental'] else 0)
        train_idx = np.arange(train_start, train_end)

        # check if seizure events in the current training set
        found_sz = found_sz or has_seizure_events(labels[train_idx])

        val_idx = np.arange(train_end, train_end + test_step_samples)
        test_idx = np.arange(train_end + test_step_samples, train_end + 2 * test_step_samples)
        yield train_idx, val_idx, test_idx


def ss_time_series_split(total_samples, test_step_samples, num_folds, args, labels=None):
    """Time-series cross-validation with minimum train data and fixed test size."""

    for fold in range(num_folds):
        train_end = total_samples - (test_step_samples * (num_folds - fold + 1))
        train_start = max(0, train_end - test_step_samples if args.mode in ['Incremental'] and fold != 0 else 0)
        train_idx = np.arange(train_start, train_end)

        val_idx = np.arange(train_end, train_end +  test_step_samples)
        test_idx = np.arange(train_end +  test_step_samples, train_end + 2 * test_step_samples)
        yield train_idx, val_idx, test_idx


def apply_butter_filter(data, fs, filter_type, cutoff, order=4):
    """
    Apply a Butterworth filter to EEG data.

    :param data: Input EEG data (2D array, channels x samples)
    :param fs: Sampling frequency
    :param filter_type: Type of filter ('highpass', 'lowpass', or 'bandstop' for notch)
    :param cutoff: Cutoff frequency (or frequencies for bandstop)
    :param order: Order of the Butterworth filter
    :return: Filtered data
    """
    nyquist = 0.5 * fs
    if filter_type == 'highpass':
        b, a = signal.butter(order, cutoff / nyquist, btype='high')
    elif filter_type == 'lowpass':
        b, a = signal.butter(order, cutoff / nyquist, btype='low')
    elif filter_type == 'bandstop':
        b, a = signal.butter(order, [cutoff[0] / nyquist, cutoff[1] / nyquist], btype='bandstop')
    else:
        raise ValueError("Unsupported filter type. Use 'highpass', 'lowpass', or 'bandstop'.")

    return signal.filtfilt(b, a, data, axis=-1).astype(np.float32)


def moving_average_filter(output: np.ndarray, fs: int, window_sec: int, threshold: float) -> np.ndarray:
    """Moving average smoothing"""
    window_size = window_sec * int(fs)
    filtered_output = np.convolve(output, np.ones(window_size), 'same') / window_size
    return filtered_output > threshold


def bayesian_smoothing(predicted_probs: np.ndarray, fs: int, window_sec: int, threshold: float) -> np.ndarray:
    """
    Perform Bayesian Smoothing on predicted probabilities.

    Parameters:
    predicted_probs (np.ndarray): Array of predicted probabilities (shape: n_samples).
    fs (int): Sampling frequency (samples per second).
    window_sec (int): Window length in seconds for smoothing.
    threshold (float): Threshold for determining the final label.

    Returns:
    np.ndarray: Array of final output labels (0 or 1) after Bayesian smoothing.
    """
    window_size = window_sec * fs
    n_samples = len(predicted_probs)

    # Initialize an array to store the final labels
    final_labels = np.zeros(n_samples)

    for i in range(n_samples):
        start_index = max(0, int(i - window_size // 2))
        end_index = min(n_samples, int(i + window_size // 2 + 1))

        # Get the probabilities within the window
        window_probs = predicted_probs[start_index:end_index]

        # Calculate cumulative products for positive and negative probabilities
        pos_probs = window_probs[window_probs > 0.5]
        neg_probs = window_probs[window_probs <= 0.5]

        if len(pos_probs) > 0 and len(neg_probs) > 0:
            cumulative_pos = np.prod(pos_probs)
            cumulative_neg = np.prod(neg_probs)

            # Calculate the logarithmic ratio
            if cumulative_neg > 0:
                log_ratio = np.log(cumulative_pos / cumulative_neg)
                final_labels[i] = 1 if log_ratio > threshold else 0
            else:
                # If no negative probabilities, consider it a positive event
                final_labels[i] = 1
        elif len(pos_probs) > 0:
            # If only positive probabilities, label as 1
            final_labels[i] = 1
        else:
            # If no positive probabilities, label as 0
            final_labels[i] = 0

    return final_labels


def create_and_append_annotation(events, mask, fs, out_file):
    """Create and append annotations."""
    annotations = epilepsy2bids.annotations.Annotations()   # Empty events
    if len(events) == 0:
        annotation = epilepsy2bids.annotations.Annotation()
        annotation["onset"] = 0
        annotation["duration"] = len(mask) / fs
        annotation["eventType"] = epilepsy2bids.annotations.EventType.bckg
        annotation["confidence"] = "n/a"
        annotation["channels"] = "n/a"
        annotation["dateTime"] = datetime.now()  # Default value
        annotation["recordingDuration"] = len(mask) / fs
        annotations.events.append(annotation)
    else:
        for event in events:
            annotation = epilepsy2bids.annotations.Annotation() # dict:{}
            annotation["onset"] = event[0]
            annotation["duration"] = event[1] - event[0]
            annotation["eventType"] = epilepsy2bids.annotations.SeizureType.sz
            annotation["confidence"] = "n/a"
            annotation["channels"] = "n/a"
            annotation["dateTime"] = datetime.now()
            annotation["recordingDuration"] = len(mask) / fs
            annotations.events.append(annotation)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    annotations.saveTsv(out_file)


def To_TSV(label, output, fs,  label_file, out_file):
    """Generate .TSV files for pred and labels, and saving results."""
    # Predict seizure events
    output = moving_average_filter(output, fs, window_sec=20, threshold=0.5)
    # output = bayesian_smoothing(output, fs, window_sec=20, threshold=0.5)
    output = timescoring.annotations.Annotation(output, fs)  # events, mask, fs
    output = timescoring.scoring.EventScoring._mergeNeighbouringEvents(output, 90)  # Merge neighboring events within 90 seconds
    label = timescoring.annotations.Annotation(label, fs)  # events, mask, fs

    # Generate annotations for output and labels
    create_and_append_annotation(output.events, output.mask, fs, out_file)
    create_and_append_annotation(label.events, label.mask, fs, label_file)


def save_annotations(annotations: epilepsy2bids.annotations.Annotations, file_path: str):
    """Save annotations to TSV file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    annotations.saveTsv(file_path)


class OversampleProcessor:
    def __init__(self, win_len, fs):
        self.win_len = win_len  # Window length in seconds
        self.fs = fs  # Sampling frequency

    def oversample_seizures(self, eeg_data, masks, pre_tolerance, post_tolerance, window_step):
        n_channels, n_samples = eeg_data.shape
        win_size = int(self.win_len * self.fs)
        step_size = int(window_step * self.fs)  # 0.05 seconds step size

        oversample_inputs = []
        oversample_labels = []

        # Identify seizure starts and ends
        sz_starts = np.where(np.diff(masks.astype(int)) == 1)[0]
        sz_ends = np.where(np.diff(masks.astype(int)) == -1)[0]

        # Handle cases where a seizure lasting until the last sample and ends at the test set
        if masks[-1] == 1:
            sz_ends = np.append(sz_ends, n_samples)

        # Step through each seizure event
        for start, end in zip(sz_starts, sz_ends):
            # Label samples from 30 seconds before the start of the seizure
            pre_start = max(0, start - int(pre_tolerance * self.fs))
            post_end = min(n_samples, end + int(post_tolerance * self.fs))
            for i in range(pre_start, post_end - win_size + 1, step_size):
                window_eeg = eeg_data[:, i:i + win_size]
                oversample_inputs.append(window_eeg)
                oversample_labels.append(1)

        # Convert to numpy arrays
        return np.array(oversample_inputs, dtype=np.float32), np.array(oversample_labels)


import gc


class CUDAPerfMonitor:
    """Context manager for timing and memory measurement"""

    def __init__(self, desc="Operation"):
        self.desc = desc
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        self._prepare()
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self._report()

    def _prepare(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def _format_memory(self, byte_size):
        for unit in ["B", "KB", "MB", "GB"]:
            if byte_size < 1024:
                return f"{byte_size:.2f} {unit}"
            byte_size /= 1024
        return f"{byte_size:.2f} TB"

    def _report(self):
        time_ms = self.start_event.elapsed_time(self.end_event)
        mem_used = self._format_memory(torch.cuda.max_memory_allocated())
        print(f"{self.desc}:")
        print(f"- Time: {time_ms:.2f} ms ({time_ms / 1000:.3f} sec)")
        print(f"- Peak Memory: {mem_used}")


# def calc_class_weight(labels_count):
#     total = np.sum(labels_count)
#     class_weight = dict()
#     num_classes = len(labels_count)
#
#     factor = 1 / num_classes
#     mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5]  # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY
#
#     for key in range(num_classes):
#         score = math.log(mu[key] * total / float(labels_count[key]))
#         class_weight[key] = score if score > 1.00 else 1.00
#         class_weight[key] = round(class_weight[key] * mu[key], 2)
#
#     class_weight = [class_weight[i] for i in range(num_classes)]
#
#     return class_weight
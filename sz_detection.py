import re
import time
from evaluate import sz_evaluate
from utils import *
from loss import *
import agents
import backbones

SECONDS_PER_HOUR = 3600

Agent = {
    'Finetune': agents.Finetune,
    'LWF': agents.LWF,
    'ER': agents.ER,
    'CLSER': agents.CLSER,
    'CLSER_SAR_LTS': agents.CLSER_SAR_LTS,
    'ESMER': agents.ESMER,
    'ESMER_SAR_LTS':agents.ESMER_SAR_LTS,
    'SAR_LTS': agents.SAR_LTS,
    'CGER':agents.CGER,
    'CGER_SAR_LTS':agents.CGER_SAR_LTS,
}


class ModelFactory:
    """
    Model factory class for creating model instances based on specified algorithms, and returning配套的 optimizers and loss functions.
    """
    @staticmethod
    def create_model(args):
        """
        Create corresponding model instances based on incoming parameters, and configure optimizers and loss functions.

        Parameters:
            args (object): Object containing model-related parameters, such as algorithm name, number of channels, window length, etc.

        Returns:
            tuple: (created model instance, optimizer instance, loss function instance).

        Raises:
            ValueError: If the incoming algorithm name is unknown, this exception is raised.
        """
        if args.algorithm == 'EEGNet':
            model = backbones.EEGNet(args.DatasetParams["channels"], args.win_len * SAMPLING_FREQUENCY).to(args.device)
        elif args.algorithm == 'EEGConformer':
            model = backbones.EEGConformer(args.DatasetParams["channels"], args.win_len).to(args.device)
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.ModelParams["lr"])
        loss_fn = FocalLoss(gamma=2.0)
        return model, optimizer, loss_fn


class DatasetUtils:
    """
    Dataset utility class for handling common operations related to datasets, such as file path extraction and data merging.
    """
    @staticmethod
    def get_edf_and_tsv_files(root_path):
        """
        Get all EDF and corresponding TSV file paths that match the format under the given root path.

        Parameters:
            root_path (str): Root path of the dataset.

        Returns:
            tuple: Contains two lists, the first list is EDF file paths, and the second list is corresponding TSV file paths.
        """
        try:
            edf_files = [str(edf.as_posix()) for edf in Path(root_path).glob("sub-*/ses-*/eeg/*_eeg.edf")]
            tsv_files = [f.replace('_eeg.edf', '_events.tsv') for f in edf_files]
            return edf_files, tsv_files
        except Exception as e:
            print(f"Error occurred while fetching files: {e}")
            return [], []

    @staticmethod
    def extract_patient_ids(edf_files):
        """
        Extract patient IDs from the EDF file path list.

        Parameters:
            edf_files (list): List of EDF file paths.

        Returns:
            list: Sorted list of patient IDs.
        """
        return sorted({Path(edf).parts[-4].split('-')[-1] for edf in edf_files})

    @staticmethod
    def get_patient_files(patient_id, edf_files, tsv_files):
        """
        Get the list of EDF and TSV file pairs corresponding to the patient based on the patient ID, and sort according to specific rules.

        Parameters:
            patient_id (str): Patient ID.
            edf_files (list): List of EDF file paths.
            tsv_files (list): List of TSV file paths.

        Returns:
            list: Sorted list of file pairs corresponding to the patient, each element is in the form (EDF file path, TSV file path).
        """
        patient_files = [(edf, tsv) for edf, tsv in zip(edf_files, tsv_files) if f"sub-{patient_id}" in edf]
        return sorted(patient_files, key=lambda x: DatasetUtils.extract_file_info(x[0]))

    @staticmethod
    def extract_file_info(file_name):
        """
        Extract sorting keys from filenames.

        Parameters:
            file_path (str): File path.

        Returns:
            Relevant information for sorting (type and content determined according to actual needs).
        """
        match = re.search(r'sub-(\d+)_ses-(\d+)_task-\w+_run-(\d+)', file_name)
        return tuple(map(int, match.groups())) if match else (-1,)


class SeizureDetectionWorkflow:
    """
    Seizure detection workflow class, coordinating the entire process of dataset processing, model training, and prediction.
    """
    def __init__(self, args):
        """
        Initialize the workflow and save the incoming parameters.

        Parameters:
            args (object): Object containing parameters related to the entire workflow, such as dataset path, model parameters, running mode, etc.
        """
        self.args = args
        self.edf_files, self.tsv_files = DatasetUtils.get_edf_and_tsv_files(self.args.DatasetParams["root_path"])
        self.patient_ids = DatasetUtils.extract_patient_ids(self.edf_files)

    def run(self):
        """
        Run the seizure detection workflow, traversing each patient to execute corresponding data processing, model training, and prediction operations.
        """
        for patient_id in self.patient_ids:
            print(f"agent: {self.args.agent}, mode: {self.args.mode}, Seizure detection for subject {patient_id}...")
            try:
                self.process_single_patient(patient_id)
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")

        print("All subjects were completed.")

    def process_single_patient(self, patient_id):
        """
        Process data for a single patient, including loading, merging, determining whether the data volume meets requirements, and executing cross-validation.

        Parameters:
            patient_id (str): Patient ID.
        """
        patient_files = DatasetUtils.get_patient_files(patient_id, self.edf_files, self.tsv_files)
        merge_patient_data = merge_patient(patient_files, self.args)    # merge_patient_session

        for patient_session_key, data in merge_patient_data.items():    # patient & session
            try:
                eegs, labels, fs = load_data(patient_session_key, self.args.DatasetParams["preprocess_path"])
                if eegs is None:
                    print(f"Merging data for patient {patient_session_key}...")
                    eegs = np.hstack([eeg.data for eeg in data['eeg']])
                    labels = np.hstack([label.mask for label in data['labels']])
                    fs = data['eeg'][0].fs

                    # Apply highpass, lowpass, notch filter at 0.5 Hz, 60 Hz, 50 Hz
                    eegs = apply_butter_filter(eegs, fs, 'highpass', 0.5)
                    eegs = apply_butter_filter(eegs, fs, 'lowpass', 60)
                    eegs = apply_butter_filter(eegs, fs, 'bandstop', [49, 51])

                    data_dir = Path(f'{self.args.DatasetParams["preprocess_path"]}/{patient_session_key}')
                    data_dir.mkdir(parents=True, exist_ok=True)
                    save_data(patient_session_key, eegs, labels, fs, data_dir)
                else:
                    print(f"Loaded data for patient {patient_session_key} from file.")

                total_hours = eegs.shape[1] / (fs * SECONDS_PER_HOUR)
                if total_hours < self.args.TSCV_MinTrainHours + 2 * self.args.TSCV_CVStepInHours:
                    print(f"Skipping patient session {patient_session_key} - data less than (min_train_hours + 2 * step_hours) hours.")
                    continue

                self.run_cross_validation_for_patient(eegs, labels, fs, patient_session_key)
            except Exception as e:
                print(f"Error processing patient session {patient_session_key}: {e}")

    def run_cross_validation_for_patient(self, eegs, labels, fs, patient_session_key):
        """
        Run time series cross-validation (TSCV) or frozen model related operations on EEG data for a single patient.

        Parameters:
            eegs (numpy.ndarray): Patient's EEG data.
            labels (numpy.ndarray): Corresponding label data.
            fs (int): Sampling frequency.
            patient_session_key (str): Identifier representing patient session.
        """
        min_train_samples = int(self.args.TSCV_MinTrainHours * SECONDS_PER_HOUR * fs)
        test_step_samples = int(self.args.TSCV_CVStepInHours * SECONDS_PER_HOUR * fs)
        total_samples = eegs.shape[1]

        if total_samples <= min_train_samples:
            print(f"Skipping patient session {patient_session_key} - not enough samples for training.")
            return

        num_folds =  max(1, (total_samples - min_train_samples) // test_step_samples - 1)

        model, optimizer, loss_fn = ModelFactory.create_model(self.args)
        agent = Agent[self.args.agent](self.args, model)

        all_test_labels = []
        all_predictions = []
        sz_in_train_test = False

        start_time = time.time()
        if self.args.mode == 'Frozen_model':
            train_idx, val_idx, test_idx = split_sz_for_frozen_model(total_samples, test_step_samples, num_folds, self.args, labels)
            if test_idx is None or len(test_idx) == 0:
                print(f"Test indices are empty for patient {patient_session_key}. Skipping this patient.")
                return

            train_data, val_data, test_data = eegs[:, train_idx], eegs[:, val_idx], eegs[:, test_idx]
            train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]

            agent.learner_train(train_data, train_labels, val_data, val_labels, fs, patient_session_key, optimizer, loss_fn, fold='frozen')
            predictions = agent.learner_predict(test_data, test_labels, fs)
            end_time = time.time()
            print(f'Forward pass time for subject {patient_session_key}: {end_time - start_time:.3f} seconds')

            out_file = Path(self.args.DatasetParams['output_path']) / f"{patient_session_key}_combined_output.tsv"
            label_file = Path(self.args.DatasetParams['output_path']) / f"{patient_session_key}_combined_label.tsv"
            To_TSV(test_labels, predictions, fs, label_file, out_file)
            sz_evaluate(self.args.DatasetParams["output_path"])

        else:
            for fold, (train_idx, val_idx, test_idx) in enumerate(
                    sz_time_series_split(total_samples, test_step_samples, num_folds, self.args, labels)):
                print(f"============== Patient {patient_session_key} Fold {fold + 1} / {num_folds} Total: {total_samples / (SECONDS_PER_HOUR * fs):.6} hours ===============")

                train_data, val_data, test_data = eegs[:, train_idx], eegs[:, val_idx], eegs[:, test_idx]
                train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]
                all_test_id = np.arange(len(labels) - (test_step_samples * (num_folds - fold)), len(labels))

                # Check if both training and test sets contain seizure events
                if has_seizure_events(train_labels) and has_seizure_events(labels[all_test_id]):
                    sz_in_train_test = True

                if not sz_in_train_test:
                    if not has_seizure_events(train_labels):
                        print(
                            f"Skipping fold {fold + 1} for {patient_session_key} - no complete seizure start and end events in train set.")
                        continue

                    if not has_seizure_events(labels[all_test_id]):
                        print(
                            f"Skipping patient session {patient_session_key} - no complete seizure start and end events found in test set.")
                        break

                # Each fold of Joint_training requires initializing the model, optimizer parameters for each patient
                if args.mode in ["Joint_training"]:
                    model, optimizer, loss_fn = ModelFactory.create_model(self.args)
                    agent = Agent[self.args.agent](self.args, model)

                agent.learner_train(train_data, train_labels, val_data, val_labels, fs, patient_session_key, optimizer, loss_fn, fold + 1)
                predictions = agent.learner_predict(test_data, test_labels, fs)
                all_test_labels.append(test_labels)
                all_predictions.append(predictions)

            end_time = time.time()
            print(f'Forward pass time for subject {patient_session_key}: {end_time - start_time:.3f} seconds')

            if len(all_test_labels) > 1:  # test fold at least >= 2, so exclude sub11, sub19...
                combined_test_labels = np.concatenate(all_test_labels)
                combined_predictions = np.concatenate(all_predictions)
                out_file = Path(self.args.DatasetParams['output_path']) / f"{patient_session_key}_combined_output.tsv"
                label_file = Path(self.args.DatasetParams['output_path']) / f"{patient_session_key}_combined_label.tsv"
                To_TSV(combined_test_labels, combined_predictions, fs, label_file, out_file)
                sz_evaluate(self.args.DatasetParams["output_path"])

            # # Save samples in the experience pool after training is completed
            # buffer_file_path = f"./buffer_samples/{patient_session_key}_fold_{fold}_buffer.pkl"
            # if hasattr(agent.buffer, 'save_buffer_to_file'):
            #     agent.buffer.save_buffer_to_file(buffer_file_path)
            #     print(f"Buffer samples saved to {buffer_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a seizure detection algorithm on an EDF file and store computed seizure annotations to a TSV."
    )
    parser.add_argument("--mode", default="Incremental", type=str,
                        help="Incremental, Joint_training, Frozen_model")
    parser.add_argument("--dataset", type=str, default="CHB-MIT",
                        choices=["CHB-MIT", "Siena", "SeizIT1"], help="dataset name.")
    parser.add_argument("--agent", default="CGER_SAR_LTS", choices=
    ["Finetune", "LWF", "ER", "CLSER", "CLSER_SAR_LTS", "ESMER", "ESMER_SAR_LTS", "SAR_LTS", "CGER", "CGER_SAR_LTS"],
                        help='Agent selection')
    parser.add_argument("--algorithm", type=str, default="EEGConformer",
                        choices=["DARNet", "EEGConformer", "EEGNet"], help="choice model to run.")
    parser.add_argument("--win_len", default=1, help="window length (second).")
    parser.add_argument("--win_step", default=1, help="window step (second), also is ultimate frequency of predictions.")
    parser.add_argument("--TSCV_MinTrainHours", default=5,  # 5, 1
                        help="The initial training set includes at least 5 hours and one seizures. set 5 hours for CHB-MIT, 1 hour for Siena")
    parser.add_argument("--TSCV_CVStepInHours", default=1,
                        help="successively adding one hour of training data and testing on the next hour.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (e.g., cuda:0, cuda:1, cuda)")
    parser.add_argument("--seed", default=42, help="random seed")
    parser.add_argument("--sampling_strategy", default="middle", choices=["random", "middle"],
                        help="Random sampling or mid-sampling of the EEG data within T seconds at intervals of 1 second.")
    args = parser.parse_args()

    # Check if CUDA is available
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1 and args.device == 'cuda':
            print(f"Using multiple GPUs: {args.device} detected, DataParallel for model training.")
        else:
            args.device = torch.device(args.device)
            print(f"Using GPU: {args.device}.")
    else:
        args.device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # avoid some error
    if args.mode in ['Frozen_model', 'Joint_training'] and args.agent not in ['Finetune']:
        raise ValueError("In Frozen_model and Joint_training mode, the agent must be set to 'Finetune'. ")

    if args.dataset == 'Siena':
        args.TSCV_MinTrainHours = 1
        args.TSCV_CVStepInHours = 0.5

    args.checkpoints = './checkpoints/' + args.algorithm

    # load .yaml config setting
    config = load_yaml_config(f'config/{args.algorithm}_{args.dataset}.yaml')

    # update config setting
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    set_random_seed(args.seed)
    workflow = SeizureDetectionWorkflow(args)
    workflow.run()

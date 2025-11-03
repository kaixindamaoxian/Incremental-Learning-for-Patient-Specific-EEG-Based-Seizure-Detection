# Incremental Learning for Patient-Specific EEG-Based Seizure Detection

This work has been accepted for publication in IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE). It presents an incremental learning framework for patient-specific EEG-based seizure detection, implemented in PyTorch. The framework supports multiple continual learning strategies and deep learning models for EEG signal classification, and provides a comprehensive set of evaluation metrics for model performance assessment.

If you find this framework useful for your research, please cite our paper:
“Incremental Learning for Patient-Specific EEG-Based Seizure Detection.”

## Dataset Download

### BIDS CHB-MIT Dataset
Download from: https://zenodo.org/records/10259996

### BIDS Siena Dataset
Download from: https://zenodo.org/records/10640762

## Project Structure

```
.
├── agents/                 # Continual learning algorithms
├── backbones/              # Deep learning model architectures
├── config/                 # Configuration files for different datasets and models
├── checkpoints/            # Saved model checkpoints
├── Trainer.py              # Base trainer class
├── utils.py                # Utility functions
├── loss.py                 # Loss functions
├── evaluate.py             # Evaluation metrics
├── sz_detection.py         # Main seizure detection workflow
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IL-seizure-detection-main
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Before running the seizure detection, configure the dataset paths in the YAML files located in the `config/` directory:
- `EEGConformer_CHB-MIT.yaml`
- `EEGConformer_Siena.yaml`
- `EEGNet_CHB-MIT.yaml`
- `EEGNet_Siena.yaml`

Update the following paths according to your local setup:
- `root_path`: Path to the downloaded BIDS dataset
- `preprocess_path`: Path to store preprocessed data
- `output_path`: Path to store output results

### Running the Detection

To run seizure detection with default settings:

```bash
python sz_detection.py
```

### Command Line Arguments

You can customize the execution with the following arguments:

```bash
python sz_detection.py \
  --mode Incremental \
  --dataset CHB-MIT \
  --agent CGER_SAR_LTS \
  --algorithm EEGConformer \
  --win_len 1 \
  --win_step 1 \
  --device cuda:0 \
  --seed 42
```

#### Arguments Explanation:

- `--mode`: Training mode
  - `Incremental`: Incremental learning (default Finetune)
  - `Joint_training`: Joint training
  - `Frozen_model`: Frozen model approach

- `--dataset`: Dataset to use
  - `CHB-MIT`: CHB-MIT dataset
  - `Siena`: Siena dataset

- `--agent`: Continual learning agent to use
  - `Finetune`: Fine-tuning approach
  - `LWF`: Learning without forgetting
  - `ER`: Experience replay
  - `CLSER`: Experience replay with complementary learning systems (CLS)
  - `CLSER_SAR_LTS`: CLSER with SAR and LTS
  - `ESMER`: Error sensitivity modulated experience replay
  - `ESMER_SAR_LTS`: ESMER with SAR and LTS
  - `CGER`: Class-Guided Experience Replay
  - `CGER_SAR_LTS`: CGER with SAR and LTS (default)
  - `SAR_LTS`: Ours

- `--algorithm`: Deep learning model architecture
  - `EEGNet`: EEGNet architecture
  - `EEGConformer`: EEG Conformer architecture (default)

- `--win_len`: Window length in seconds (default: 1)
- `--win_step`: Window step in seconds (default: 1)
- `--device`: Computing device (default: cuda:0)
- `--seed`: Random seed for reproducibility (default: 42)

## Model Architectures

### EEGNet
Implementation of the EEGNet architecture for EEG signal classification.

### EEGConformer
Implementation of the EEG Conformer architecture, combining convolutional and transformer layers for EEG signal processing.

## Evaluation

The framework automatically evaluates the model performance using:
- Sample-level scoring (sensitivity, precision, F1-score, FP rate)
- Event-level scoring (sensitivity, precision, F1-score, FP rate)

## Continual Learning Agents

The framework implements various continual learning approaches to handle the sequential nature of EEG data:

1. **Finetune**: Traditional fine-tuning approach
2. **LWF**: Learning without Forgetting
3. **ER**: Experience Replay
4. **CLSER**: Clustered Sample Replay
5. **ESMER**: Efficient Sample Management Experience Replay
6. **CGER**: Class-Guided Experience Replay
7. **SAR_LTS**: Sample Attractiveness and Representativeness with Local Temporal Similarity

Each approach handles the catastrophic forgetting problem differently while learning from new data sequences.

## Results

The framework outputs TSV files with seizure predictions that can be used for further analysis and comparison with ground truth annotations.

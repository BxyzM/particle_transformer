# Particle Transformer - Master Thesis Extension

This repository is based on the official [Particle Transformer](https://arxiv.org/abs/2202.03772) implementation, extended for master thesis research on binary jet classification tasks.

## Master Thesis Focus

This work extends the original Particle Transformer framework with (in progress):

- **Binary Classification Tasks**: TTbar vs QCD, WZ vs QCD, HToCC vs QCD
- **Evaluation Tool**: ROC curve analysis, AUC calculation, and background rejection metrics

## Project Structure

```
├── data/                    # Dataset configurations
├── training/               # Training outputs and model checkpoints
│   └── JetClassBinary/    # Binary classification experiments
│       ├── TTbar_vs_QCD/
│       ├── WZ_vs_QCD/
│       └── HToCC_vs_QCD/
├── plotters/              # Analysis and visualization tools
│   ├── auc_calc_binary_general.py  # General binary ROC/AUC calculator
├── models/                # Pre-trained model weights
└── networks/             # Model architectures
```

## New Features

### Binary Classification Evaluation

**General Binary ROC Calculator** (`plotters/auc_calc_binary_general.py`):
- Auto-detects binary classification type from prediction files
- Supports multiple signal processes (TTbar, WZ, HToCC)
- Calculates AUC and background rejection at 50% signal efficiency
- Generates ROC curves with proper metrics

**Usage:**
```bash
python ./plotters/auc_calc_binary_general.py /path/to/predict_output
```

The script automatically detects the classification task based on the ROOT files in the directory and produces:
- AUC score
- Background rejection at 50% TPR
- ROC curve plots saved to `predict_output/plots/`

## Getting Started

### Prerequisites

```bash
pip install 'weaver-core>=0.4'
pip install uproot awkward
pip install matplotlib mplhep
pip install scikit-learn
```

### Training Binary Classification Models

**TTbar vs QCD:**
```bash
./train_JetClassBinary.sh ParT TTbar_vs_QCD kin --gpus 0 --batch-size 512
```

**WZ vs QCD:**
```bash
./train_JetClassBinary.sh ParT WZ_vs_QCD kin --gpus 0 --batch-size 512
```

**HToCC vs QCD:**
```bash
./train_JetClassBinary.sh ParT HToCC_vs_QCD kin --gpus 0 --batch-size 512
```


### Evaluating Results

Generate ROC curves and calculate metrics:

```bash
python ./plotters/auc_calc_binary_general.py /path/to/training/predict_output
```

## Thesis Research Questions


## Results Structure

Training outputs are organized as:
```
training/JetClassBinary/{SIGNAL}_vs_QCD/{FEATURES}/ParT/
└── samples{N}_epochs{E}/
    └── {TIMESTAMP}_example_ParticleTransformer_ranger_lr{LR}_batch{BS}/
        ├── predict_output/
        │   ├── pred_{SIGNAL}_vs_QCD_{SIGNAL}.root
        │   ├── pred_{SIGNAL}_vs_QCD_QCD.root
        │   └── plots/
        │       └── roc_{SIGNAL}_binary.png
        └── net_best_epoch_state.pt
```

## Original Particle Transformer

This work builds upon the original Particle Transformer framework. For the original implementation and JetClass dataset, see:

- **Paper**: [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772)
- **Dataset**: [JetClass on Zenodo](https://zenodo.org/record/6619768)
- **Original Repo**: [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer)



## License

This project maintains the same license as the original Particle Transformer repository.

## Contact

For questions regarding the thesis extension, please contact: uuwdo@student.kit.edu

For questions about the original ParT implementation, refer to the [original repository](https://github.com/jet-universe/particle_transformer).

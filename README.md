# CAPE: Pre-training Epidemic Time Series Forecasters with Compartmental Prototypes

[[Paper]](https://arxiv.org/abs/2502.03393)

A foundation model for epidemic time series forecasting that combines compartmental (SIR-family) dynamics with transformer-based token prediction. CAPE uses multi-mask ensemble inference over diverse compartmental structures to produce robust forecasts without parametric fine-tuning.

## Setup

**Requirements:** Python 3.10, CUDA-capable GPU

```bash
conda create -n cape python=3.10 -y
conda activate cape
pip install -r requirements.txt
```

**Checkpoints:** Download from [Google Drive](https://drive.google.com/drive/folders/1UiGo4go9sCCIcGCPITbfWvRojSSgaJk6?usp=sharing) and place the `checkpoints/` folder in the project root:

```
CAPE/
├── checkpoints/
│   └── pretraining/
│       ├── next_token/checkpoint.pth            # Stage 1: next-token pretraining
│       ├── next_token_pretrain_v5/checkpoint.pth # Stage 1 (full)
│       └── second_stage_pretrain_v2/             # Stage 2 (used for evaluation)
│           └── checkpoint.pth
├── data/
│   └── tycho_US.pt                              # Project Tycho US disease data
├── src/                                         # Evaluation pipeline
├── model/                                       # Model definitions
└── ...
```

## Evaluation

Run evaluation across diseases:

```bash
# Evaluate all modes (chronos2, moirai, moment, zeroshot) on default diseases
bash src/run_online_comparison.sh

# Evaluate a specific disease
bash src/run_online_comparison.sh measle

# Evaluate specific modes only
bash src/run_online_comparison.sh measle --modes zeroshot

# Evaluate on all in-domain or out-of-domain diseases
bash src/run_online_comparison.sh --indomain
bash src/run_online_comparison.sh --outdomain
bash src/run_online_comparison.sh --all_diseases
```

Summarize results and generate radar plots:

```bash
python src/summarize_results.py --plot_radar
```

Results are saved to `src/results_online/`.

### Diseases

| In-domain (8) | Out-of-domain (9) |
|---|---|
| Pertussis, Varicella, Tuberculosis, Measles, TyphoidFever, Mumps, Diphtheria, ScarletFever | Smallpox, Influenza, Pneumonia, AcutePoliomyelitis, MeningococcalMeningitis, Gonorrhea, HepatitisA, HepatitisB, Rubella |

### Metrics

- **MSE / MAE** — Standard forecasting error
- **Outbreak Recall** — Fraction of true outbreaks detected
- **Alert Sensitivity** — Sensitivity of early warning alerts
- **Peak Underestimate Rate** — How often peaks are underestimated
- **Rising Phase MAE** — Error during epidemic growth phases

## Pretraining

To retrain CAPE from scratch:

```bash
# Stage 1: Next-token pretraining on synthetic compartmental data
bash run_pretrain.sh

# Stage 2: Second-stage pretraining on real epidemic data
bash src/run_second_stage_pretrain.sh
```

## Project Structure

```
├── model/
│   ├── CAPE_Compartmental.py  # Core CAPE model with compartmental dynamics
│   ├── chronos.py             # Chronos & Moirai baseline wrappers
│   └── Moment.py              # MOMENT baseline wrapper
├── src/
│   ├── run_online_comparison.sh  # Main evaluation script
│   ├── strategy_online.py        # Rolling-fold evaluation pipeline
│   ├── evaluate_utils.py         # Shared utilities (data loading, prediction collection)
│   ├── ensemble_strategies.py    # 14 non-learnable ensemble strategies
│   └── summarize_results.py      # Results aggregation & radar plots
├── utils/
│   ├── epidemic_metrics.py    # Epidemic-specific evaluation metrics
│   └── metrics.py             # Standard forecasting metrics
├── dataset/                   # Data loading & synthetic generation
├── data/tycho_US.pt           # Project Tycho US disease surveillance data
└── epirecipe/                 # EpiRecipe synthetic data generation toolkit
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{liu2025pretrainingepidemictimeseries,
      title={Pre-training Epidemic Time Series Forecasters with Compartmental Prototypes},
      author={Zewen Liu and Juntong Ni and Max S. Y. Lau and Wei Jin},
      year={2025},
      eprint={2502.03393},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.03393},
}
```


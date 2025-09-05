# ambit-lpm: Longitudinal Phenotype Modeling for Rare Disease Diagnosis

## ToDo

- [ ] Reimplement using pytorch?
- [ ] Add tests for trajectory validators
- [ ] Improve trajectory sampling strategy
- [ ] Track how many times each possible trajectory is sampled for a patient
- [ ] Track loss at each month endpoint
- [ ] Add ASM codes to trajectories
- [ ] Learning curves (performance vs. number of positive patients)
- [ ] Add simulated trajectories?
- [ ] Learning rate warmup?
- [ ] Filter out irrelevant ICD10 codes?
- [ ] Could we train for longer?

## Setup

### 1. Create a conda environment

```{bash}
conda create --name ambit-lpm python=3.11
conda activate ambit-lpm
```

### 2. Install (with dependencies)

Note that this setup.py currently installs a specific version of tensorflow. This can probably be removed to allow for more flexibility.

```{bash}
# for base functionality
pip install --upgrade .

# for demos and scripts
pip install --upgrade ".[demos,scripts]"
```

## Usage

### Preprocessing

```{bash}
python scripts/risknet/preprocess.py
```

### Training

```{bash}
python scripts/risknet/train.py
```

## Possible Analyses

1. Can we somehow look at the estimated TPR/FPR and then quantify the benefit of implementing this type of approach for patient screening - i.e., if we were to screen patients based on the predicted risk, how many more patients would we catch earlier? What cost would we incur due to genetic testing in false positives? Would this be outweighed by a reduction in per-patient healthcare costs due to earlier detection?

## Notes

### Trajectory Sampling

- RiskNet currently samples balanced bathces of trajectories for pos/neg patients
- Epochs are tied to the number of positive patients

## Resources

- [CancerRiskNet](https://github.com/BrunakSanderLabs/CancerRiskNet)
- [A deep learning algorithm to predict risk of pancreatic cancer from disease trajectories](https://www.nature.com/articles/s41591-023-02332-5)
- [SHEPHERD](https://github.com/mims-harvard/SHEPHERD/tree/main)
- [Few shot learning for phenotype-driven diagnosis of patients with rare genetic diseases](https://www.medrxiv.org/content/10.1101/2022.12.07.22283238v2)

# DataScienceSpotify – Music Hits

Course project for the Kaggle competition **music-hits-2026**: predict the probability that a song is a hit (`music_hit`).
The metric is **ROC-AUC**.

## Notebooks

Run them in order. Each one builds on the output of the one before.

| Notebook | What it does |
| --- | --- |
| `01_eda.ipynb` | Explores the data |
| `02_validation_split.ipynb` | Makes a fixed 80/20 validation split plus 5 CV folds and computes single-feature baselines |
| `03_feature_engineering.ipynb` | Builds the features, including missing flags and out-of-fold target encoding |
| `04_logistic_regression.ipynb` | Logistic regression |
| `05_random_forest.ipynb` | Random forest |
| `06_gradient_boosting.ipynb` | Gradient boosting (HistGradientBoosting) |
| `07_ft_transformer.ipynb` | FT-Transformer for tabular data |
| `08_ensemble.ipynb` | Compares and combines the models and writes the submission |

Shared code lives in `features.py` (feature building) and `utils.py` (loading data, baseline check, saving predictions).

## Data

`train.csv`, `test.csv` and `sample_submission.csv` come from Kaggle and sit in the project root.

## Results

Everything the notebooks produce is saved in `results/`:

- `splits.csv`: which songs are in the validation set and which CV fold each one belongs to
- `baseline_scores.csv`: validation AUC of every single feature (the score to beat)
- `predictions/`: validation and test predictions for each model
- `submission.csv`: the final Kaggle submission

| Model | Validation AUC |
| --- | --- |
| Best single feature (`artist_followers`) | 0.875 |
| Logistic regression | 0.9595 |
| Random forest | 0.9662 |
| FT-Transformer | 0.9677 |
| **Gradient boosting** | **0.9699** |

The ensembles did not beat gradient boosting by a meaningful margin, so the submission uses gradient boosting on its own.

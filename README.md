
(https://soundofmolecules.com)
markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Small-Data Binary Classifier Diagnostic

A diagnostic and modeling tool for the classic small-data problem: a handful of
process factors, ~100 experiments, a Yes/No outcome, and a model that "never
works."

With roughly 100 data points and 10 factors, the right approach is **not** a
neural network. This tool runs appropriately simple, regularized models with
proper cross-validation, ranks which factors actually carry signal, and gives
an honest verdict on whether the outcome is even predictable from the data you
collected.

## What it does

1. Loads the data and reports shape and class balance.
2. Flags the small-data regime (samples-per-factor, class imbalance).
3. Runs four models with stratified k-fold cross-validation:
   - Regularized logistic regression (the right default at this scale)
   - Small random forest (captures nonlinear factor interactions)
   - Gradient-boosted trees (often best on tabular data)
   - A majority-class baseline (the bar any real model must clear)
4. Reports ROC-AUC and balanced accuracy with their cross-validated spread.
5. Ranks which factors carry signal using permutation importance.
6. Delivers a plain-language verdict: is the signal there, or is this a
   design-of-experiments problem?

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`

Install:

```
pip install numpy pandas scikit-learn matplotlib
```

## Usage

```
python small_data_classifier.py data.csv --outcome outcome
```

Options:

- `--outcome` — name of the Yes/No outcome column (default: `outcome`)
- `--folds` — number of cross-validation folds (default: `5`; automatically
  reduced if the minority class is too small to support that many)

## Input format

One row per experiment. One column is the binary outcome; the rest are the
factors you varied.

| Column type | Handling |
|---|---|
| Numeric factors | Median-imputed and standardized automatically |
| Text factors (e.g. buffer type) | One-hot encoded automatically |
| Outcome column | `Yes/No`, `1/0`, `True/False`, `pass/fail` all accepted |
| Missing values | Imputed automatically (median for numeric, most-frequent for text) |

A sample `data.csv` is included as a format template: 100 rows, 7 numeric and
3 categorical factors, a balanced `Yes/No` outcome, and a few missing cells to
show the imputation working.

## Reading the results

- **ROC-AUC** is the headline metric. Judge by this, not raw accuracy, since
  accuracy is misleading when classes are imbalanced.
- **Balanced accuracy** is a sanity check that the model treats both classes
  fairly.
- **Factor importance** tells you which knobs actually matter, so you can stop
  varying the ones that don't.

Rough guide to the AUC verdict:

| AUC | Interpretation |
|---|---|
| ≥ 0.75 | Clear signal; a working classifier is achievable |
| 0.65 – 0.75 | Weak but real; more data and fewer noise factors will help |
| 0.55 – 0.65 | Marginal; likely a design-of-experiments problem |
| ≈ 0.50 | No usable signal; revisit the experiment design |

With ~100 points a single number is noisy. Trust the cross-validated mean and
its ± spread, not one lucky train/test split.

## Files

- `small_data_classifier.py` — the tool
- `data.csv` — sample data / format template
- `README.md` — this file

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
MIT License

Copyright (c) 2026 [Emily Zhou / SoundofMolecules.Com]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
(https://soundofmolecules.com)


# Frozen Days Linear Regression

This project fits a linear regression model to data on the number of frozen days per year. It uses both the closed-form solution and gradient descent (implemented with PyTorch) and visualizes the data, the loss curve, and predictions.

## Overview

The script performs the following steps:

1. Load a CSV file with columns:
   - `year`
   - `days` (number of frozen days in that year)
2. Plot the raw data (`data_plot.jpg`).
3. Normalize the input years to the range [0, 1].
4. Compute the closed-form least squares solution for linear regression.
5. Use gradient descent with PyTorch to learn the same model and record the loss over iterations.
6. Plot the loss curve (`loss_plot.jpg`).
7. Use the learned model to predict the number of frozen days for a given test year (e.g., 2024).
8. Print simple interpretations of the learned slope and limitations of the model.

## File

- `frozen_days_regression.py` — main script with loading, normalization, linear regression, gradient descent, plotting, and prediction.

## Requirements

- Python 3
- numpy
- pandas
- matplotlib
- torch

You can install the requirements with:

    pip install numpy pandas matplotlib torch

## How to Run

Prepare a CSV file, for example `frozen_days.csv`, with at least the following columns:

- `year`
- `days`

Example:

    year,days
    1855,118
    1860,120
    1870,110
    ...

Then run:

    python3 frozen_days_regression.py frozen_days.csv

The script will:

- Print the normalized design matrix (Q2).
- Print the closed-form weights (Q3).
- Run gradient descent and print intermediate weights and the learning rate (Q4).
- Save `data_plot.jpg` (year vs. frozen days).
- Save `loss_plot.jpg` (iteration vs. loss).
- Print the prediction for a future year (Q5) and simple interpretation and limitations (Q6, Q7).

## Notes

- Normalization is applied only to the input years; the target variable (days) is kept in original scale.
- The gradient descent implementation uses mean squared error as the loss.
- The chosen learning rate is tuned so that gradient descent converges close to the closed-form solution.

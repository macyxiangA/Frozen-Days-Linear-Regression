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

- frozen_days_regression.py — main script
- ice_data.csv — dataset containing `year` and `days`
- toy.csv — small sample dataset for testing
- data_plot.jpg — plot of frozen days vs. year (auto-generated)
- loss_plot.jpg — gradient-descent loss plot (auto-generated)
- README.md — documentation

## Requirements

- Python 3
- numpy
- pandas
- matplotlib
- torch

You can install the requirements with:

    pip install numpy pandas matplotlib torch

## How to Run

Run the script with any CSV file:

    python3 frozen_days_regression.py ice_data.csv

or:

    python3 frozen_days_regression.py toy.csv

After running, the script will:

- Print the normalized design matrix (X_normalized)
- Print closed-form regression weights
- Print gradient-descent weights every 20 iterations
- Output a prediction for year 2024
- Output the sign of the slope (increasing or decreasing trend)
- Output a simple explanation and model limitations
- Save:
  - data_plot.jpg
  - loss_plot.jpg

## What the Script Does

1. Loads year and frozen-day counts from CSV.  
2. Plots the original data and saves the figure.  
3. Normalizes year values to the range [0, 1].  
4. Computes the linear regression solution using the closed-form formula.  
5. Runs gradient descent using PyTorch for 200 iterations and tracks loss.  
6. Saves a loss-curve plot.  
7. Predicts frozen days for 2024.  
8. Reports the trend direction (positive, negative, or none).  
9. Prints why long-term predictions may not be reliable.

## Notes

- Normalization is applied only to the input years; the target variable (days) is kept in original scale.
- The gradient descent implementation uses mean squared error as the loss.
- The chosen learning rate is tuned so that gradient descent converges close to the closed-form solution.

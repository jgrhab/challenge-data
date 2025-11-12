# Stock Return Prediction by QRT (2020)

https://challengedata.ens.fr/challenges/23

## Overview

The notebook uses [LightGBM](https://lightgbm.readthedocs.io) to predict the target class.

The notebook achieves a score (accuracy) of **52.41%**.
The corresponding leaderboard rank at time of submission (March 10, 2025) is 16.

## Notebook parameters and structure

The notebook can be run in two modes: testing and submission. 
In testing mode, a fraction of training data is withheld to act as test set and provide an evaluation of the final model, 
while in submission mode, the competition test set is used to generate the submission file.
In both modes, the training data is first partitioned into training and validation sets, and the model is tuned
and evaluated using the validation set, before being retraining on the entire dataset to predict the test

## Feature engineering

Features are added following the example of the benchmark. 
These consist of statistics (mean and variance) of the returns and volumes taken over groups (sector, etc.) 
for individual days (as in the benchmark) as well as over a window of the first five days (i.e. over $1 \leq t \leq 5$ for each row).
The model input is restricted to the two days preceeding the target date, meaning that only the features with 
$t = 1$ and $t = 2$ are passed to the model. 
The rolling statistics still provide context for a longer time frame.
No categorical feature is included, as these negatively impacted performance in testing. 
The result is 29 features.

## Data augmentation

Data augmentation is used to increase the number of training examples by considering past
return values as successive targets. 
This is done by shifting each row of data to the left. 
For instance, in the first shift, `RET_1` becomes the new target, `RET_2` becomes `RET_1`, and so on.
The data is shifted in this manner as many times as possible, with the number of shifts
depending on the number of time-steps used for the rolling statistics (five) and as inputs (two). 
This results in 15 augmentation steps. 
Augmenting both the training and test data results in close to nine million training examples.

The values of `RET_t` are converted to binary categories by comparing to the daily median, as is done in the benchmark. 
This preserves the balance per day of the target variable from the original dataset, and is also used for the predictions.

## Feature selection and hyperparameter tuning

Feature selection was done using [SHAP](https://shap.readthedocs.io) to assess the impact of individual features.
Parameter tuning was done partily by hand (by looking at the shape of the training curve) and partly using [Optuna](https://optuna.org/).

This process is not included in the notebook as the iterative nature is difficult to convey without dedicating a considerable amount of space to it.

## Ideas for improvement

- engineer more meaningful features
- further reduce overfitting (e.g. using [DART](https://arxiv.org/abs/1505.01866))
- use clustering on predictions to identify patterns

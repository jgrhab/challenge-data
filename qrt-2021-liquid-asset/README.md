# Reconstruction of Liquid Asset Performance, by QRT (2021)

https://challengedata.ens.fr/challenges/44/

## Overview 

The notebook uses an averaging process to combine multiple regressors to predict the targets.

The notebook achieves a score (accuracy) of **74.81%**.
The corresponding leaderboard rank at time of submission (May 6, 2025) is 13/279.

## Individual models

Various models are built and evaluated by sampling validation data from the training set.
These include linear regression models (with L1 regularization), kernel principal component
regression, partial least squares regression, and gradient boosted trees.
The individual models are tuned to achieve the best possible score with respect to the 
challenge metric (custom weighted accuracy) and evaluated on a per target basis.
This reveals a discrepency both between the targets (some being easier to predict than others)
and between the models, which perform differently depending on the target.

## Averaging model

The individual models are combined by averaging their output.
The purpose of this averaging process is twofold:
1. Reduce variance for the points where most predictions are good.
Since the goal is to predict the sign of each target point, prediction for target values 
close to zero tend to be unstable even for models with low overall error.
This is partly remedied by the averaging process, which acts as regularizer for the predictions.
2. Capture the shape of certain targets which is inaccessible to certain models.
Because all 100 targets are differents, some models are better suited for certain targets
than others. Linear Regression for instance can only linearly separate the target classes,
which is good enough for most targets but might fail for certain cases where a more complex
boundary reflects the underlying truth. Other more complicated models can capture this
more complex relationship between inputs and targets, but often at the cost of worse 
overall accuracy due to a higher sensitivity to variation in the fitting process.
The idea is that averaging should allow a form of weighted voting to determine the sign
of the final prediction, where linear models with low certainty will predict small values
while non-linear models can make more confident predictions (larger values) to outweigh
the low-confidence ones.

This approach is justified a posteriori since the averaging model bests all individual models
globally and attains performance similar to that of the best model for each target locally.

This project uses random feauturization to detect wildfires in satellite imagery. 

The field of combining satellite imagery with machine learning (SIML) has vast implications across
many different applications. From estimation of socioeconomic climate in data-poor regions to global
predictions of environmental conditions, SIML holds tremendous promise for remote assessments of
various tasks. While there are numerous advantages, this field is computationally expensive and often
requires special resources for training and evaluation. To address this computational need, a machine
learning architecture that leverages random convolutional features for a one-time task agnostic encoding
has been proposed to generalize across various prediction tasks. This featurization then uses regression
to train and evaluate the model given ground truth data for a specific task, achieving state-of-the-art
performance with significantly reduced computational cost. This code compares the results and cost
analysis of two models, (1) 2-layer convolutional neural network and (2) random convolutional features
and regression, on satellite imagery to predict the presence of wildfires at a specific location.

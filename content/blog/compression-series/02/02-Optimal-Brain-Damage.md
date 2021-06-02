---
date: "2021-03-29"
draft: false
excerpt: Notes on the paper Optimal Brain Damage by Yann LeCun, John S. Denker and Sara A. Solla
subtitle: ""
title: Optimal Brain Damage
weight: 2
---

[Check out the Paper](https://papers.nips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)

**Key Takeaway:** The basic idea is to use second-order derivative information to make a tradeoff between network complexity and training set error.

## Introduction 

The Authors used information-theoretic ideas to derive a class of practical and nearly optimal schemes for adapting the size of a neural network. By removing the unimportant weights from a network, several improvements can be expected:

- Better Generalisation
- Fewer Training samples required
- Improved speed of learning and/or classification

Most successful applications of NN learning to real-world problems have been achieved using highly structured networks of large sizes. As applications become more complex, the networks would presumably become even larger and more structured. The authors <span style="color:orange">**propose Optimal Brain Damage (OBD)**</span> for reducing the size of a learning network by selectively deleting weights.

The basic idea of OBD is that it is possible to take a perfectly reasonable network, delete half (or more) of the weights and wind up with a network that works just as well, *or better*. It is known from theory and experience that, for a fixed amount of training data, networks with too many weights do not generalize well. On the other hand, networks with too few weights will not have enough power to represent the data accurately. <span style="color:orange"> **The best generalization is obtained by trading off the training error and the network complexity.**</span>

## Possible Approaches

One such method could be to minimize a cost function composed of two terms, the ordinary training error, plus some measure of the network complexity. 

But a simpler strategy exists which doesn't involve priori or heurisitc information i.e. to delete parameters with small <span style="color:orange">**"saliency"**</span> (those whose deletion has the least effect on the training error). Other things being equal, small-magnitude parameters will have the least saliency, so a reasonable initial strategy is to train the network and delete small-magnitude parameters. After deletion, the network is retrained. Two drawbacks of these techniques are that they require fine-tuning of the "pruning" coefficients to avoid catastrophic effects, and also that the learning process is significantly slowed down.

## Optimal Brain Damage

Objective functions play a central role, therefore it is more than reasonable to define the saliency of a parameter to be the change in objective function caused by deleting some parameter. It wouldbe prohibitively laborious to evaluate the saliency directly from this definition, i.e. by temporarily deleting each parameter and re-evaluating the objective function

However, it is possible to construct a local model of the error function and analytically predict the effect of perturbating the parameter vector. The objective function $E$ is approximated using Taylor Series. Here, the perturbation $ \mathcal{L} $ of the parameter vector will change the objective function by :

$$
\delta{E} = \sum_{i} g_i  \, \delta{u}_i + \frac{1}{2}\sum_{i} h_{ii} \delta {u_{i}}^2 + \frac{1}{2} \sum_{i \neq j} h_{ij} \delta{u}_{i} \delta{u}_{j} + O(||\mathcal{L}||^{3})
$$

The goal is to find a set of parameters whose deletion will cause the least increase of $ E $. The problem is computationally expensive in general cases, therefore we introduce some approximations. After using diagonal, extremal and local minimum approximation, this equation simplifies to:

$$
\delta{E} = \frac{1}{2} \sum_{i} h_{ii} \delta{u}^2_{j}
$$

## The OBD Recipe

1. Choose a reasonable network architecture
2. Train the network until a reasonable solution is obtained
3. Compute the second derivatives $ h_{kk} $ for each parameter
4. Compute the saliences for each parameter: $ s_k = h_{kk}u^2_k/2 $
5. Sort the parameters the saliency amd delete some low-saliency parameters.
6. Iterate to step 2
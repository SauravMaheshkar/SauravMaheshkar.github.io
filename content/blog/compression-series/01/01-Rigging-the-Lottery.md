---
date: "2021-03-31"
draft: false
excerpt: Notes on the paper Rigging the Lottery Making All Tickets Winners by Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro and Erich Elsen
subtitle: ""
title: Rigging the Lottery
weight: 1
---

[Check out the Paper](https://arxiv.org/abs/1911.11134)

**Key Takeaway:** The authors introduce RigL - an algorithm for training sparse neural networks while maintaining memory and computational cost proportional to density of the network. RigL achieves higher quality than all previous techniques for a given computational cost. RigL can find more accurate models than the current best dense-to-sparse training algorithms. Provide insight as to why allowing the topology of non-zero weights to change over the course of training aids optimization.

## Introduction

Many applications require sparse neural networks due to space or inference time restrictions. There is a large body of work on training dense networks to yield sparse networks for inference, but this **limits the size of the largest trainable sparse model to that of the largest trainable dense model**. In this paper, the authors introduce a method to train sparse neural networks with a fixed parameter count annd a fixed computational cost throughout training without sacrificing accuracy.

This method updates the topology of the sparse network during training by using parameter magnitudes and infrequent gradient calculations.

## Current Limitations

Currently, the most accurate sparse models are obtained with techniques that require, at a minimum, the cost of training a dense model in terms of memory and FLOPs. This paradigm has 2 main limitations:

- The maximum size of sparse models is limited to the largest dense model that can be trained. Even if sparse models are more parameter efficient, we can't use pruning to train models that are larger and more accurate than the largest possible dense models.
- It is inefficient. Large amounts of computation must be performed for parameters that are zero valued or that will be zero during inference. Additionally, it remains unknown if the performance of the current best pruning algorithms is an upper bouond on the quality of sparse models.

The Lottery Ticket Hypothesis hypothesized that if we can find a sparse neural network with iterative pruning, then we can train that sparse neural network from scratch, to the same level of accuracy, by starting from the initial conditions.

## The RigL Method

![alt](/img/rigl/rigl.png)

RigL starts with a random sparse network, and at regularly spaced intervals it removes a fraction of connections based on their magnitudes and activates new ones using instantaneous gradient information. There are 4 main parts of the algorithm:

1. Sparsity Distribution
2. Update Schedule
3. Drop Criterion
4. Grow Criterion

### Notation

- Given a dataset $D$ with individual samples $x_i$ and targets  $y_i$ , the aim is to minimize the loss function $\sum_{i} L (f_{\theta}(x_i), y_i)$ , where $f_{\theta}(.)$ is a neural network with parameters $\theta \in \mathbb{R}^{N}$.
- Parameters of the $l^{th}$ layer are denoted with $\theta^{l}$ which is a length $N^{l}$ vector.
- A sparse layer keeps only a fraction $s^{l} \in (0,1)$ of its connections and parameterized with vector $\theta^{l}$ of length $(1 - s^{l})N^{l}$.
- The overall sparsity of the network is defined as the ratio of zeros to the total parameter count $S = \frac{\sum_{l} s^{l} N^{l} }{N}$

### Sparsity Distribution

The following 3 strategies were considered:

1. **Uniform:** The sparsity $s^{l}$ of each individual layer is equal to the total sparsity $S$.
2. **Erdos-Renyi:** $s^{l}$ scales with $1 - \frac{n^{l-1} + n^{l}}{n^{l-1}*n^{l}}$ , where $n^{l}$ denotes the number of neurons in layer $l$.
3. **Erdos-Renyi-Kernel(ERK):** Modifies the original Erdos-Renyi formulation by including the kernel dimensions in the scaling factors.  In other words, the number of parameters of the sparse convolutional layers are scaled proportional to equation - (1) .Here, $w^{l}$ and $h^{l}$ are the width and height of the $l^{th}$ convolutional kernel.

$$1 - \frac{n^{l-1} + n^{l} + w^{l} + h^l}{n^{l-1}*n^{l}*w^{l}*h^{l}} \, \, \, \, \, \,(1)$$

### Update Schedule

The update schedule is defined by the following parameters:

1. $\Delta T$: The number of iterations between sparse connectivity updates
2. $T_{end}$ : The iteration at which to stop updating the sparse connectivity
3. $\alpha$ : The initial fraction of connections updated
4. $f_{decay}$ : a function invoked every $\Delta T$ iterations until $T_{end}$, possibly decaying the fraction of updated connections over time.

$$f_{decay}(t \, ; \alpha, T_{end} ) = \frac{\alpha}{2} \left ( 1 + cos(\frac{t \pi}{T_{end}}) \right )$$

### Drop Criterion

Every $\Delta {T}$, steps we drop the connections given by

$$ArgTopK(-|\theta^{l}|, (1 - s^{l})N^{l})$$

where $ArgTopK(v,k)$ gives the indices of the top-$k$ elements of vector $v$.

### Grow Criterion

We grow the connections with highest magnitude gradients. Newly activated connections are **initialized to zero** and therefore don't affect the output of the network.
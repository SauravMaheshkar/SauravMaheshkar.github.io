---
date: "2020-09-27"
draft: false
excerpt: Notes on the paper Learning Nonlinear Functions Using Regularized Greedy Forest by Rie Johnson
categories:
  - Paper Summaries
subtitle: ""
title: Regularized Greedy Forest
weight: 1
---

[Check out the Paper](https://arxiv.org/pdf/1109.0887.pdf)

**Key Takeaway:** A new proposed method which learns decision forests via fully-corrective regularized greedy search using the underlying forest structure by defining regularizers that explicitly take advantage of individual tree structures.


## Introduction

![](/img/regular-greedy-forest.jpg)

A popular method to solve the problem of *learning non-linear functions from data* is through decision tree learning , which has an important advantage for handling heterogeneous data with ease when different features come from different sources.

However, a disadvantage of decision tree learning is that it does not generally achieve the most accurate
prediction performance, when compared to other methods. A remedy for this problem is through **boosting** where one builds an additive model of decision trees by sequentially building trees one by one. In general “boosted decision trees” is regarded as the most effective off-the-shelf nonlinear learning method for a wide range of application problems.

In the boosted tree approach, one considers an additive model over multiple decision trees, and thus, we will
refer to the resulting function as a ***decision forest***.

Due to the practical importance of boosted decision trees in applications, it is natural to ask whether one can design a more direct procedure that specifically learns decision forests without using a black-box decision tree learner under the wrapper. The purpose of doing so is that by directly taking advantage of the underlying tree
structure, we shall be able to design a more effective algorithm for learning the final nonlinear decision forest.

---

We consider the problem of learning a single nonlinear function $h(x)$ on some input vector $x = [ x_{[1]}, ..., x_{[d]} ] \in \mathbb{R}^d$ from a set of training examples. In supervised learning, we are given a set of input vectors $X = [x_1, ..., x_n]$ with labels $Y = [y_1, ..., y_m]$ (here <b>m may not equal to n</b>). Our training goal is to find a nonlinear prediction function $\hat{h}(x)$ from a function class $ H $ that minimizes a risk function. 

$$\hat{h} = \min_{h \in H} L(h(X), Y)$$

$H$ → is a pre-defined nonlinear function class $h(X) = [h(x_1), ..., h(x_n)]$

$L$  → General Loss Function

## Notation (The Complex Part 😅)

- A Forest is an ensemble of multiple decision trees $T_1, T_2, ..., T_k$
- Each tree edge $e$ is associated with a variable $k_e$ and threshold $t_e$ and denotes a decision
- Mathematically, each node $v$ of the forest is associated with a decision rule of the form

$$ b_{v}(x) = \prod_{j} I(x[i_j] \leq t_{i_{j}}) \prod_{k} I (x[i_k] > t_{i_{k}}) $$

- $h_{F}(x) = \sum_{v \, \in F} \alpha_vb_v(x)$ , where $\alpha$ is the weight assigned to a node ( equal to zero for any internal node )

Thus, regularized loss is of the form

$$ Q(F) = L(h_{F}(X), Y) + R(h_{F}) $$

## Algorithmic Framework

Since, the exact optimum solution is difficult to find, we **greedily select the basis functions and optimize the weights**. It has two main components:

- Fix the weights, and change the structure of the forest (which changes basis functions) so that the loss is reduced the most. 
- Fix the structure of the forest, and change the weights so that loss is minimized.

## Tree-structured regularization

For any additive models over leaf sample nodes only, there always exist equivalent models over all the nodes of the same tree that produce the same output.

Our basic idea is that it is natural to give the same regularization penalty to all equivalent models defined on the same tree topology. One way to define a regularizer that satisfies this condition is to choose
a model of some desirable properties as the unique representation for all the equivalent models and define the regularization penalty based on this unique representation.

The following penalty functions could be choosed:

- $L_2$ regularization
- Minimum-penalty regularization
- Min-penalty regularization with sum-to-zero sibling constraints
---
date: "2021-02-19"
draft: false
excerpt: Notes on the paper Every Model Learned by Gradient Descent Is Approximately a Kernel Machine by Pedro Domingos
categories:
  - Paper Summaries
subtitle: ""
title: Every Model is a Kernel Machine
weight: 1
---

## Introduction 

We all know how poorly understood deep learning is, in recent times fields such as Explainability and Interpretability have emerged highlighting the black box that are deep nets. In contrast, kernel machines are based on a well-developed mathematical theory but their performance lacks behind that of deep networks. The most used algorithm for training deep networks is undoubtedly Gradient Descent. In this paper the author shows that every model learnt using this method, regardless of its architecture, is approximately equivalent tp a kernel machine 🤔. This kernel learns the similarity of the model at two data points in the neighborhood of the path taken by the model parameters during learning. Kernel machines store a subset of the training data points and match them to the query using the kernel and thus deep network weights are nothing but a superposition of the training data points in the kernel's feature space 🤨 🤨.  

This contrasts with the traditional school of thought 🧐 that deep learning is a method for discovering representations from data. 


## Path Kernels 

A kernel machine is of the form:

$$ y = g( \sum_{i} a_iK(x,x_i) + b ) $$

where 

* $ x $ is a query data point, 
* The sum is over all training data points $ x_i $
* $ g $ is an optional nonlinearity
* $ a_i $ and $ b $ are learned parameters
* The kernel $ K $ measures the similarity of its arguments

For instance in supervised learning, $ a_i $ is typically a linear function of $ \hat{ y_{i} } $ (the known output of $ x_i $). Kernel machines, are also known as support vector machines. 

But whether a representable function is learned or not depends on the learning algorithm. Most of us are familiar with the following:

$$ w_{s+1} = w_{s} - \epsilon\nabla_{w}L(w_{s}) $$

where:

* $ L = \sum_{i} L(\hat{ y_{i} }, y_{i}) $ is the loss function
* $ w $ are the model parameters
* $ \epsilon $ is the learning rate

This process terminates when the gradient becomes zero and we reach the saddle point. 

The author terms **path kernels** to be those kernel machines which result from gradient descent. Assuming the learning rate to be infinitesimally small, the path kernel between any two arbitrary data points is simply the dot product of the model's gradients at the two points over the path taken by the parameters during gradient descent:

$$ K(x,x^{'}) = \int_{c(t)} \nabla_{w}y(x).\nabla_{w}y(x^{'}) dt $$

where $ c(t) $ is the path.  Intuitively, the path kernel measures how similarly the model at the two data points varies during learning. The more similar the variation for $ x $ and $ x_{'} $, the higher the weight of $ x_{'} $ in predicting $y$.



## Some Definitions

1. **Tangent Kernel:** The Tangent Kernel associated with function $f_w(x)$ and the parameter vector $v$ is:

$$
K_{f,v}^{g}(x, x^{'}) = \nabla_{w}f_w(x) . \nabla_wf_w(x^{'}) - (I)
$$

2. **Path Kernel:** The path kernel associated with function $ f_w(x) $ and curve $c(t)$ in parameter space is:

$$
K_{f,c}^{p}(x, x^{'}) =  \int_{c(t)}^{} K_{f,v}^{g}(x, x^{'}) dt - (II)
$$


## The Proof


As per Gradient Descent:

$$ 
w_{s+1} = w_{s} - \epsilon\nabla_{w}L(w_{s}) 
$$

$$
\frac{ w_{s+1} - w_{s} }{ \epsilon } = - \nabla_{w}L(w_{s}) 
$$ 

This becomes a differential equation of the form(Also known as <span style="color:blue">Gradient Flow</span>.):

$$
\frac{dw(t)}{dt} = - \nabla_{w}L(w(t)) - (1)
$$

Then for any differentiable function of the weights $y$ (<span style="color:green">Using Chain Rule</span>)

$$
\frac{dy}{dt} = \sum_{j=1}^{d} \frac{ \partial{y}}{ \partial{w_j} } \frac{\partial{w_j}}{\partial{t}}
$$

Here, $d$ is the number of parameters. Replacing $dw_j/dt$, by it's value from ${eq}^{n} \, (1)$, we get:

$$
\frac{dy}{dt} = \sum_{j=1}^{d} \frac{ \partial{y}}{ \partial{w_j} } \left (- \frac{\partial{L}}{\partial{w_j}} \right)
$$

Applying <span style="color:green">Chain Rule</span> and <span style="color:green">additivity of the loss</span>, we get:

$$
\frac{dy}{dt} = \sum_{j=1}^{d} \frac{ \partial{y}}{ \partial{w_j} } \left (- \sum_{i = 1}^{m} \frac{\partial{L}}{\partial{y_i}} \frac{\partial{y_i}}{\partial{w_j}} \right)
$$

Rearranging the terms a little bit, we get:

$$
\frac{dy}{dt} = \sum_{i = 1}^{m}  \frac{\partial{L}}{\partial{y_i}}  \left (- \sum_{j=1}^{d} \frac{ \partial{y}}{ \partial{w_j} } \frac{\partial{y_i}}{\partial{w_j}} \right)
$$

Let $ L^{'}(y_{i}^{*}, y_{i}) = \partial{L} / \partial{y_i} $ (<span style="color:green">The loss derivative for the ith output</span>). Applying this and **Definition(I)** (<span style="color:green">Tangent Kernel</span>):

$$
\frac{dy}{dt} = \sum_{i = 1}^{m}  L^{'}(y_{i}^{*}, y_{i}) K_{f,w(t)}^{g}(x, x^{'})
$$

If $ y_0 $ is the initial model, prior to the gradient descent. Then for the final model $y$;

$$
\lim_{\epsilon \rightarrow 0} y =  y_0 - \int_{c(t)} \sum_{i = 1}^{m}  L^{'}(y_{i}^{*}, y_{i}) K_{f,w(t)}^{g}(x, x^{'})
$$

where $c(t)$ is the path taken by the parameters during gradient descent. Multiplying and dividing by $\int_{c(t)} K_{f,w(t)}^{g}(x, x^{'}) dt$, we get:

$$
\lim_{\epsilon \rightarrow 0} y =  y_0 - \int_{c(t)} \sum_{i = 1}^{m} \left( \frac{ \int_{c(t)} K_{f,w(t)}^{g}(x, x^{'}) dt L^{'}(y_{i}^{*}, y_{i})dt}{\int_{c(t)} K_{f,w(t)}^{g}(x, x^{'}) dt} \right) K_{f,w(t)}^{g}(x, x^{'}) 
$$

Let the term in the braces $()$ be $\bar{L^{'}}$ i.e., <span style="color:green">the average loss derivative weighted by similarity to x</span>. Applying this and definition (II), we get:

$$
\lim_{\epsilon \rightarrow 0} y =  y_0 - \int_{c(t)} \sum_{i = 1}^{m} \bar{L^{'}(y_{i}^{*}, y_{i}) K_{f,c}^{p}(x, x^{'})}
$$

Thus, finally we get

$$
\lim_{\epsilon \rightarrow 0} y = \sum_{i=1}^{m} a_i K(x, x_i) + b
$$

where,

* $ b = y_0 $
* $ a = -\bar{L^{'}}(y_{i}^{*}, y_i) $
* $ K(x, x_i) = K_{f,c}^{p} (x, x_i) $ 
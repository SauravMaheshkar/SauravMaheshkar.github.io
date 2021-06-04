---
author: Saurav Maheshkar
categories:
- compression
- python
- tensorflow
date: "2021-03-03"
draft: false
excerpt: Minimal Reproducibility Study of (https://arxiv.org/abs/1911.05248). Experiments with Compression of Deep Neural Networks
layout: single-sidebar
links:
- icon: door-open
  icon_pack: fas
  name: Web Application
  url: https://share.streamlit.io/sauravmaheshkar/compressed-dnns-forget/web-app/app.py
- icon: github
  icon_pack: fab
  name: code
  url: https://github.com/SauravMaheshkar/Compressed-DNNs-Forget/
- icon: file-alt
  icon_pack: fas
  name: Weights and Biases Report
  url: https://wandb.ai/sauravmaheshkar/exploring-bias-and-compression/reports/What-Do-Compressed-Deep-Neural-Networks-Forget---Vmlldzo1NzA0NDY
title: Compressed DNNs Forget
---

![](https://github.com/SauravMaheshkar/Compressed-DNNs-Forget/raw/main/assets/Pruning.png)

## Reproducibility Study

Current state-of-the-art models are famously huge and over-parameterized––in fact, they contain way more parameters than the number of data points in the dataset. But in many ways, over-parameterization is behind the success of modern-day deep learning. Think about something like Switch Transformer having a trillion parameters or Vision Transformer-Huge having 632M parameters. These models require enormous amounts of computation and memory and that not only increases the infrastructure costs, but also makes deployment to resource-constrained environments such as mobile phones or smart devices challenging. 

With this push towards bigger and deeper models comes the competing need for fast deployment and efficiency. One tactic that solves some of this give-and-take is compression. Specifically, practitioners have started focusing on model compression methods like pruning and quantization and have proved that training a larger model followed by pruning beats training a smaller model from scratch. Gale et al., 2019 beautifully demonstrated that unstructured, sparse architectures learned through pruning cannot be trained from scratch to the same test set performance as a model trained with pruning as part of the optimization process.

Several techniques for building efficient AI have been proposed over the past few years such as:

* Automated Design (Auto-ML)
* Knowledge Distillation
* Quantization
* Tensor Decomposition
* Pruning

In this project I dig into pruning, paying special attention to a recent paper called [**"What Do Compressed Deep Neural Networks Forget?"**](https://arxiv.org/pdf/1911.05248.pdf) by Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin and Andrea Frome.

## 📚 Questions raised by the paper

* How can networks with radically different representations and number of parameters have comparable top-level metrics ? One possibility is that test-set accuracy is simply not a precise enough measure to capture how compression impacts the generalization properties of the model. 

* Are certain types of examples or classes disproportionately impacted by model compression techniques like pruning and quantization?

* What makes performance on certain subsets of the dataset far more sensitive to varying model capacity?

* How does compression impact model sensitivity to certain types of distributional shifts such as image corruptions and natural adversarial examples ?

---

An understanding of the trade-offs incurred by model compression is critical when quantized or compressed deep neural networks are used for sensitive tasks where bad predictions are especially problematic–areas like facial recognition, health care diagnostics, or self-driving cars. 

Additionally, results on CelebA show that PIE over-indexes on protected attributes like gender and age, suggesting that compression may amplify existing algorithmic bias. For sensitive tasks, the introduction of pruning may be at odds with fairness objectives to avoid disparate treatment of protected attributes and/or the need to guarantee a level of recall for certain classes. Compression techniques are already widely used in sensitive domains like health care in order to fulfill tight resource constraints of deployment.
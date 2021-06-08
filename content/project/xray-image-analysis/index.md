---
author: Saurav Maheshkar
categories:
- python
- tensorflow
date: "2021-04-10"
draft: false
excerpt: 🩺 Analysing Deep Learning Methods for X Ray Image Analysis
featured: true
layout: single
links:
- icon: book
  icon_pack: fas
  name: Notebook
  url: https://deepnote.com/@saurav-maheshkar/X-Ray-Image-Classification-M7ID1ABnTqysUd7USuIaUw
- icon: microchip
  icon_pack: fas
  name: Web Application
  url: https://share.streamlit.io/sauravmaheshkar/x-ray-image-classification/streamlit/demo/app.py
- icon: door-open
  icon_pack: fas
  name: Github Repository
  url: https://github.com/SauravMaheshkar/X-Ray-Image-Classification/
subtitle: ""
title: X Ray Image Analysis
---

![](https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/Banner%20Image.png?raw=true)

In this project, I went over the entire pipeline of creating a Binary Image Classifier using Tensorflow. I covered, all aspects of the pipeline such as experimenting with different network architectures and comparing metrics, pruning and quantization of the model for faster inference and finally two methods of deploying such models. I trained a Convolutional Neural Network(Efficient Net) to recognize chest X-rays of people with pneumonia and then created a interactive web application.

The Dataset for this project, titled "Chest X-Ray Images (Pneumonia)" was taken from Kaggle and was uploaded by Paul Mooney. The dataset was organized into 3 folders ( train, test, val ) and contained subfolders for each image category (Pneumonia/Normal). There were 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). These images were then converted into TFRecords format, zipped and then upload as a Shared Dataset to the Deepnote Project.

## The Model 👷‍♀️

### 🏗 Transfer Learning

First things first; Transfer learning(TL) is not a machine learning model or technique; it is rather a **"design methodology"** within machine learning. The general idea of transfer learning is to use knowledge learned from tasks for which a lot of labelled data is available in settings where only little labelled data is available. Creating labelled data is expensive, so optimally leveraging existing datasets is key.

In a traditional machine learning model, the primary goal is to generalise to unseen data based on patterns learned from the training data. With transfer learning, you attempt to kickstart this generalisation process by starting from patterns that have been learned for a different task. Essentially, instead of starting the learning process from a (often randomly initialised) blank sheet, you start from patterns that have been learned to solve a different task.

Convolutional Neural Networks' features are more generic in early layers and more original-dataset-specific in later layers. Thus, we often use these as a backbone / starting point while creating new models. A common practice is to make these base models non-trainable and just learn the later layers. You might think that this will decrease the performance, but as we'll see from training. Transfer Learning is still a viable option.

### ⚖️ EfficientNetB0

![](https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/effnet.png?raw=true)

**Convolutional neural networks (CNNs)** are commonly developed at a fixed resource cost, and then scaled up in order to achieve better accuracy when more resources are made available. For example, ResNet can be scaled up from ResNet-18 to ResNet-200 by increasing the number of layers. The conventional practice for model scaling is to arbitrarily increase the CNN depth or width, or to use larger input image resolution for training and evaluation. While these methods do improve accuracy, they usually require tedious manual tuning, and still often yield suboptimal performance. Instead, the authors of ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (ICML 2019)"](https://arxiv.org/abs/1905.11946) found a more principled method to scale up a CNN to obtain better accuracy and efficiency.

They proposed a novel model scaling method that uses a simple yet highly effective compound coefficient to scale up CNNs in a more structured manner. Unlike conventional approaches that arbitrarily scale network dimensions, such as width, depth and resolution, their method uniformly scales each dimension with a fixed set of scaling coefficients. The resulting models named EfficientNets, superpassed state-of-the-art accuracy with up to 10x better efficiency (smaller and faster).

## 🚀 Deployment

As deep learning is infiltrating most industries, learning how to deploy models into production is becoming an extremely important skill to master. In this project, I went over two methods to deploy the trained models

1. Web Application built using Streamlit (Uses Quantized TFLite model)
2. Using Tensorflow Serving (Uses SavedModel Format)

### Streamlit Web Application

![](https://github.com/SauravMaheshkar/X-Ray-Image-Classification/blob/main/assets/xray-app.gif?raw=true)

Streamlit is probably the fastest way to share data models. Rather than Flask or Django which requires the developer to know a little bit about HTML, CSS and JavaScript, using Streamlit you can build responsive applications using just python. It has an extremely simple API and has a free sharing platform, where you can deploy your local model to a website in just a few clicks. The code for this app can be found in the accompanying Github Repository. I have also shared a [Docker Image](https://github.com/SauravMaheshkar/X-Ray-Image-Classification/packages/745856) for anyone to be able to clone and run the model locally using just 2 commands.

### Tensorflow Serving

TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data. I have upload a [custom serving image](https://github.com/SauravMaheshkar/X-Ray-Image-Classification/packages/746338) (with the model mounted) built on top of the tensorflow/serving image. You can clone this image expose a port and call it using a REST API.
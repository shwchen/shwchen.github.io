---
layout: post
title: "Improvements on Uncertainty Quantification for Node Classification via Distance-Based Regularization"
date: "2024-05-29"
slug: "improve-uncertain-quantification"
description: "Deep neural networks have achieved significant success in the last decades, but they are not well-calibrated and often produce unreliable predictions. A large number of literature relies on uncertainty quantification to evaluate the reliability of a learning model, which is particularly important for applications of out-ofdistribution (OOD) detection and misclassification detection. We are interested in uncertainty quantification for interdependent node-level classification. We start our analysis based on graph posterior networks (GPNs) that optimize the uncertainty cross-entropy (UCE)-based loss function. We describe the theoretical limitations of the widely-used UCE loss. To alleviate the identified drawbacks, we propose a distance-based regularization that encourages clustered OOD nodes to remain clustered in the latent space. We conduct extensive comparison experiments on eight standard datasets and demonstrate that the proposed regularization outperforms the state-of-the-art in both OOD detection and misclassification detection."
category: 
  - Machine Learning (cs.LG)
  - Machine Learning (stat.ML)
tags:  # tags will also be used as html meta keywords.
  - Node Classification
  - Misclassification Detection
  - Out-of-Distribution
  - NeurIPS'23
show_meta: true
comments: true
mathjax: true
gistembed: true
published: true
noindex: false
nofollow: false
hide_printmsg: false  # hide QR code, permalink block while printing.
summaryfeed: false  # show post summary or full post in RSS feed.
---

**Authors:** [Russell Hart](https://www.semanticscholar.org/author/Russell-Hart/2061118286), [Linlin Yu](https://www.semanticscholar.org/author/Linlin-Yu/2266365133), [Yifei Lou](https://www.semanticscholar.org/author/Yifei-Lou/2266238849), [Feng Chen](https://www.semanticscholar.org/author/Feng-Chen/2266265893) \
**Venue:** 37th Conference on Neural Information Processing Systems (NeurIPS 2023 Poster). \
**Links:** [ArXiv](https://arxiv.org/abs/2311.05795), [OpenReview](https://openreview.net/forum?id=MUzdCW2hC6). \
**Code:** [GitHub](https://github.com/neoques/Graph-Posterior-Network) \
**BibTex:** 
~~~r
@article{Hart2023ImprovementsOU,
  title={Improvements on Uncertainty Quantification for Node Classification via Distance-Based Regularization},
  author={Russell Hart and Linlin Yu and Yifei Lou and Feng Chen},
  journal={ArXiv},
  year={2023},
  volume={abs/2311.05795},
  url={https://api.semanticscholar.org/CorpusID:265128811}
}
~~~

# 1 Preliminary

## 1.1 Uncertainty

When talking about predictive uncertainty of a DNN, it is referring to indicating when the DNN's predictions are likely incorrect. There are two main types of uncertainty, which are typically used for OOD and misclassification detection, respectively:

### 1.1.1 Epistemic Uncertainty

>*Epistemic*: relating to knowledge or to the dgree of its validation.

This type of uncertainty is due to the lack of knowledge about unseen data.

### 1.1.2 Aleatoric Uncertainty

>*Aleatoric*: depending on the throw of a dice or on chance; random.

This type of uncertainty is caused by the inherent complexity of the data, which cannot be reduced by increasing the training data, including sources of noise such as homoscedastic or heteroscedastic noise.

**Homoscedastic noise** refers to a situation where the variance of the noise is constant across the input space. In other words, the amount of noise or uncertainty is the same regardless of the input values.

**Heteroscedastic noise**, on the other hand, refers to a situation where the variance of the noise changes depending on the input values. The amount of noise or uncertainty is not constant but rather varies across the input space.

For example, in a regression problem, if the noise has the same variance regardless of the input variable, it would be considered homoscedastic. But if the noise variance increases or decreases depending on the input variable, it would be considered heteroscedastic.

### 1.1.3 Graph Posterior Network (GPN)

#### 1.1.3.1 Axiomatic Approach

>Axiom 1: A node's prediction without network effects should only depend on its own features. A node with features more different from training features should be assigned higher uncertainty.

>Axiom 2: All else being equal, if a node's prediction without network effects is more epistemically certain, then its neighbors' predictions with network effects should become more epistemically certain.

>Axiom 3: All else being equal, a node's prediction with network effects should have higher aleatoric uncertainty if its neighbors's prediction without network effects have high aleatoric uncertainty. Further, a node prediction with network effects should have higher aleatoric uncertainty if its neighbors' predictions without network effects are more conflicting.

#### 1.1.3.2 Bayesian Update

Standard Bayesian Update

Aleatoric: $$y\sim Cat(p)$$; \
Epistemic: $$p\sim Dir(\alpha^{post})$$ where $$\alpha^{post} = \alpha^{prior} + \beta$$

Input-dependent Bayesian Update

Aleatoric: $$y^{(u)}\sim Cat(p^{(u)})$$; \
Epistemic: $$p^{(u)}\sim Dir(\alpha^{post,(u)})$$ where $$\alpha^{post,(u)} = \alpha^{prior} + \beta^{ft.(u)}$$

Bayesian Update for Interdependent Input

Aleatoric: $$y^{(u)}\sim Cat(p^{(u)})$$; \
Epistemic: $$p^{(u)}\sim Dir(\alpha^{post,(u)})$$ where $$\alpha^{post,(u)} = \alpha^{prior} + \beta^{agg.(u)}$$

#### 1.1.3.3 Intuitive Understanding

Imagine you have a social network where each person can have one of several hobbies. Some people’s hobbies are known, while others are not. The goal is to predict the hobbies of the unknown ones. GPN uses the known hobbies and the connections between people to predict the unknown hobbies. It also indicates how confident it is about each prediction. For instance, if someone is connected to many people with the same hobby, GPN will be more confident in predicting that hobby for them.

### 1.1.4 Uncertainty Cross-Entropy (UCE)

# 2 Motivation

The uncertainty estimation for classifying interdependent nodes in attributed graph data is under-explored. Unlike other data such as images and tabulars, if a node in a graph is somehow perturbed, the distorted information will spread through the node's neighbors owing to the message-passing aggregation scheme. 

The authors focus on the node classification tasks with great potential to generalize to others with interdependent inputs. They further theoretically analyze the limitations of GPN at OOD detection when minimizing UCE, then propose a distance-based regularization accordingly.

# 3 Method

# 4 Experiment

# 5 Summary


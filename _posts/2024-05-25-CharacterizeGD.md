---
layout: post
title: "Characterizing Graph Datasets for Node Classification: Homophily–Heterophily Dichotomy and Beyond"
date: "2024-05-05"
slug: "characterize-graph-datasets"
description: "Homophily describes the tendency for similar nodes to connect, while heterophily involves dissimilar nodes, challenging GNNs. Current homophily measures have drawbacks. This work introduces 'adjusted homophily', which meets more desirable properties, and proposes 'label informativeness' (LI) to better distinguish heterophily types. LI aligns well with GNN performance, proving useful for characterizing graph structure."
category: 
  - Social and Information Networks (cs.SI)
  - Discrete Mathematics (cs.DM)
  - Machine Learning (cs.LG)
  - Probability (math.PR)
tags:  # tags will also be used as html meta keywords.
  - Node Classification
  - Homophily
  - heterophily
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

**Authors:** [Oleg Platonov](https://arxiv.org/search/cs?searchtype=author&query=Platonov,+O), [Denis Kuznedelev](https://arxiv.org/search/cs?searchtype=author&query=Kuznedelev,+D), [Artem Babenko](https://arxiv.org/search/cs?searchtype=author&query=Babenko,+A), [Liudmila Prokhorenkova](https://arxiv.org/search/cs?searchtype=author&query=Prokhorenkova,+L). \
**Venue:** 37th Conference on Neural Information Processing Systems (NeurIPS 2023). \
**Links:** [ArXiv](https://arxiv.org/abs/2209.06177), [OpenReview](https://openreview.net/forum?id=D4GLZkTphJ). \
**Code:** Source not available. [DGL implementation](https://docs.dgl.ai/en/2.0.x/generated/dgl.node_label_informativeness.html). \
**BibTex:** 
~~~r 
@inproceedings{Platonov2022CharacterizingGD,
  title={Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond},
  author={Oleg Platonov and Denis Kuznedelev and Artem Babenko and Liudmila Prokhorenkova},
  booktitle={Neural Information Processing Systems},
  year={2022},
  url={https://api.semanticscholar.org/CorpusID:256808725}
} 
~~~

# Introduction

## Preliminary

### Homophily
>Edges tend to connect *similar* nodes. For instance, users in social networks tend to connect to users with similar interests, and papers in citation networks mostly cite works from the same research area.

Some popular Homophily Measures: \
**Edge Homophily:** computes the fraction of edges that connect nodes of the same class. 

$$
  h_{edge} = \frac{|\{\{u, v\} \in E: y_u = y_v\}|}{|E|}
$$

**Node Homophily:** computes the fraction of neighbors of same class for all nodes.

$$
  h_{node} = \frac{1}{n}\sum_{v\in V}\frac{|\{\{u, v\} \in N(v): y_u = y_v\}|}{d(v)}
$$

**Class Homophily:** measures excess homophily compared to a null model where edges are independent of the labels.

$$
  h_{class} = \frac{1}{C-1}\sum_{k=1}^{C}\left[ \frac{\sum_{v:y_v=k}|\{u\in N(v):y_u=y_v\}|}{\sum_{v:y_v=k}d(v)} - \frac{n_k}{n} \right]_{+}
$$

Node and Edge Homophily are sensitive to number of classes and their balance. Class Homophily addresses the issue but only consider positive deviation from $$\frac{n_k}{n}$$ (neglecting heterophilous patterns) and does not consider variation of node degrees.

### Heterophily

>Edges tend to connect *dissimilar* nodes. For instance, in social networks, fraudsters rarely connect to other fraudsters, while in dating networks, edges often connect the opposite genders.

### Constant Baseline

>Requires a measure being unbiased towards particular numbers of classes or their size balance.

## Motivation

Several Homophily Measures used prevalently do not satisfy the Constant Baseline requirement and may disagree with each other across different graph datasets. The authors claim that `Adjusted Homophily is a better choice as it meets most of the requirements.`{:.yelhglt}

Heterophilous graph datasets can have various connectivity patterns and some of them are easier for GNNs than others. `The authors proposed a new graph property, named Label Informativeness (LI), which characterizes how much information the neighbor's label provides about the node's label, to complement Adjusted Homophily by distinguishing different homophilous patterns.`{:.yelhglt} Empirical evidence testifies that LI better explains GNNs' performance than Homophily Measures.

# Method and Theory

# Experiment and Analysis

# Thought and Discussion


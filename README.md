# METRON: Metabolic Dynamic Perception Kolmogorov-Arnold Network for Biological Age Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of the paper **"METRON: Metabolic Dynamic Perception Kolmogorov-Arnold Network for Biological Age Estimation"**.

![Teaser Image](./framework overview.png)
Figure 1: Proposed Architecture of METRON Framework for Biological Age Estimation. (a) The overall workflow. Input data is measured from human metabolomics and fed into METRON for biological age estimation. (b) The detailed architecture of METRON. The input data is first normalized and then processed sequentially by a Metabolite Interaction Perception Module and a Group-Rational Kolmogorov-Arnold Network (GR-KAN), linked by residual connections. A final prediction head regresses the learned features to a BA value. (c) The internal structure of the Metabolite Interaction Perception Module. This module employs a deformable attention mechanism, where a lightweight network ($\theta_{off}$) learns sampling offsets ($\Delta p$) to adaptively select key and value features from the input ($Z_e$). (d) The GR-KAN layer. Unlike a standard MLP, GR-KAN uses learnable, group-wise rational functions ($R_i(x)$) that are applied to inputs before a linear transformation. The colors indicate that the parameters of the rational functions are shared within distinct input groups. Illustration created with elements from Servier Medical Art, licensed under CC BY 3.0.

## 🚀 Abstract
Biological age provides a more direct reflection of physiological status than chronological age, serving as a vital measure to evaluate health risks and aging interventions. While steroid metabolomics offers rich information for exploring aging mechanisms, the complex and non-linear interactions within metabolic networks remain challenging in modeling. Here, we propose and describe METRON as a deep learning framework to predict biological ages from steroid metabolomics. Specifically, a Metabolite Interaction Perception Module (MIPM) is proposed to capture the interactions. Subsequently, a Group-Rational Kolmogorov-Arnold Network is also integrated to capture intricate dependencies and enhance the representation capability. We demonstrate that METRON achieves promising performance as compared to other machine learning and deep learning methods. Beyond performance, METRON offers interpretability by recovering the established markers such as Dehydroepiandrosterone (DHEA) and identifying 17-hydroxyprogesterone (17-OH-P4) as the key signature linked to hypothalamic-pituitary-adrenal axis dynamics. These results support the capacity of METRON not only to estimate biological age but also to uncover underappreciated metabolic drivers behind aging.

## 📢 News
* **[2026-01-18]**: The codes are released.

## 📂 Data Preparation
Dataset can be downloaded from "Biological age prediction using a DNN model based on pathways of steroidogenesis" (https://www.science.org/doi/10.1126/sciadv.adt2624)

## 📄 License
This project is released under the MIT License.

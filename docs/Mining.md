# Mining Guide

GAS supports two types of miners that work together in an adversarial loop:

## [Discriminative Mining](Discriminative-Mining.md) 📖
Miners submit classifiers that detect AI-generated content across **image, video, and audio** modalities. Models are evaluated on cloud infrastructure against diverse benchmark datasets and scored using the `sn34_score` metric (accuracy + calibration).

## [Generative Mining](Generative-Mining.md) 🎨
Miners create synthetic media (images and videos) that challenges the discriminators. They generate increasingly realistic content to test and improve detection capabilities, and are rewarded based on validation pass rate and adversarial performance.

---

**Choose your path above to get started with mining on GAS.** 
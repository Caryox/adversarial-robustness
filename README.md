# Evaluation of the Effectiveness using Ensemble Adversarial Training in Combination with Conventional Input Rectification and APE-GAN Architecture in Combination with Few2Decide against Adversarial Perturbation Attacks​

This repo contains our code moduls and data for our paper with the same name. 

## Abstract

Image classification is applied in many applications nowadays; thus, its algorithms are constantly being honed to become more accurate and efficient. Along with that development come several security concerns. The number of proven powerful attack methods against classification algorithms is growing to such a degree that the awareness about vulnerabilities of neural networks must be raised. In this paper, we aim to find the best defense methods against state-of-the-art attacks by combining proved defenses in recent years. Firstly, we implemented two combinations of defense techniques: Ensemble Adversarial Training in Combination with Conventional Input Rectification and APE-GAN architecture in Combination with Few2Decide. These two combinations were applied on MNIST and CIFAR-10 datasets, then attacked by FGSM, PGD, DeepFool and Carlini & Wagner’s attack. For the Ensemble experiments, we extended the amount of used data in comparison to the original setups to achieve better plausibility. We found out that Ensemble Adversarial Training in Combination with Conventional Input Rectification is robust against adversarial perturbation attacks in general. In contrast, combining the APE-GAN architecture with Few2Decide technique does not improve the defense ability significantly. Furthermore, the results of many defense techniques could not be reconstructed or explained due to lack of clarity in papers. More transparency in implementation is needed to research and to solve the adversarial perturbation problem. 

## Table of content

### Data
- [MNIST](https://github.com/Caryox/adversial-robustness/tree/main/data/MNIST) 
- [CIFAR-10](https://github.com/Caryox/adversial-robustness/tree/main/data/cifar-10-batches-py)

### Main Modules
- [APE-GAN](https://github.com/Caryox/adversial-robustness/tree/main/src/APE_GAN)
- [Ensemble Models](https://github.com/Caryox/adversial-robustness/tree/main/src/Models) 

### Example Images
Our ensemble modul
![Ensemble](https://github.com/Caryox/adversial-robustness/blob/fc70b735438bafb5d275c1ca33cc52e5209739bc/data/Ensemble%20Conventional%20Rectification%20V4_Updated.jpg)
Image processing
![Image processing](https://github.com/Caryox/adversial-robustness/blob/fc70b735438bafb5d275c1ca33cc52e5209739bc/data/example_image_processing.jpg)

Adversarial Detection
![grafik](https://user-images.githubusercontent.com/56730144/187691315-aea67a9a-a745-4a40-bc92-5d006822a81b.png)
Try to calculate best threshold t for seperate adversarials and legitimate samples by measuring the maximum L1-Distance of the SoftMax-Layer between the baseline CNN ensemble and the feature squeezed CNN ensembles


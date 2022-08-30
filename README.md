# Evaluation of the Effectiveness using Ensemble Adversarial Training in Combination with Conventional Input Rectification and APE-GAN Architecture in Combination with Few2Decide against Adversarial Perturbation Attacks​

This repository contains all code modules for the paper with the same name.
Read the paper here

## Abstract

Image classification is applied in many applications nowadays; thus, its algorithms are constantly being honed to become more accurate and efficient. Along with that development come several security concerns. The number of proven powerful attack methods against classification algorithms is growing to such a degree that the awareness about vulnerabilities of neural networks must be raised. In this paper, we aim to find the best defense methods against state-of-the-art attacks by combining proved defenses in recent years. Firstly, we implemented two combinations of defense techniques: Ensemble Adversarial Training in Combination with Conventional Input Rectification and APE-GAN architecture in Combination with Few2Decide. These two combinations were applied on MNIST and CIFAR-10 datasets, then attacked by FGSM, PGD, DeepFool and Carlini & Wagner’s attack. For the Ensemble experiments, we extended the amount of used data in comparison to the original setups to achieve better plausibility. We found out that Ensemble Adversarial Training in Combination with Conventional Input Rectification is robust against adversarial perturbation attacks in general. In contrast, combining the APE-GAN architecture with Few2Decide technique does not improve the defense ability significantly. Furthermore, the results of many defense techniques could not be reconstructed or explained due to lack of clarity in papers. More transparency in implementation is needed to research and to solve the adversarial perturbation problem. 

## Table of content






### Linux and Mac Users

- run the setup script `./setup.sh` or `sh setup.sh`

### Windows Users

- run the setup script `.\setup.ps1`

## Pictures of our experiments
Image smoothing examples 


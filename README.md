# Evaluation of the Effectiveness using Ensemble Adversarial Training in Combination with Conventional Input Rectification and APE-GAN Architecture in Combination with Few2Decide against Adversarial Perturbation Attacks​

This repo contains our code moduls and data for our paper with the same name. 

## Abstract

Image classification is applied in many applications nowadays; thus, its algorithms are constantly being honed to become more accurate and efficient. Along with that development come several security concerns. The number of proven powerful attack methods against classification algorithms is growing to such a degree that the awareness about vulnerabilities of neural networks must be raised. In this paper, we aim to find the best defense methods against state-of-the-art attacks by combining proved defenses in recent years. Firstly, we implemented two combinations of defense techniques: Ensemble Adversarial Training in Combination with Conventional Input Rectification and APE-GAN architecture in Combination with Few2Decide. These two combinations were applied on MNIST and CIFAR-10 datasets, then attacked by FGSM, PGD, DeepFool and Carlini & Wagner’s attack. For the Ensemble experiments, we extended the amount of used data in comparison to the original setups to achieve better plausibility. We found out that Ensemble Adversarial Training in Combination with Conventional Input Rectification is robust against adversarial perturbation attacks in general. In contrast, combining the APE-GAN architecture with Few2Decide technique does not improve the defense ability significantly. Furthermore, the results of many defense techniques could not be reconstructed or explained due to lack of clarity in papers. More transparency in implementation is needed to research and to solve the adversarial perturbation problem. 


### Data

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) 
- [CIFAR-10 Dataset ](https://www.cs.toronto.edu/~kriz/cifar.html)

### Main Modules

- [APE-GAN](https://github.com/Caryox/adversial-robustness/tree/main/src/APE_GAN) 
- [Ensemble Models](https://github.com/Caryox/adversial-robustness/tree/main/src/Models) 

### Implemented Papers
#### Defenses
- [ENSEMBLE ADVERSARIAL TRAINING](https://arxiv.org/abs/1705.07204) and Conventional Input Rectification [1](http://arxiv.org/abs/1705.10686) [2](http://arxiv.org/abs/1511.04508)
- [APE-GAN: Adversarial Perturbation Elimination with GAN](https://arxiv.org/abs/1707.05474) and [Few2Decide](https://link.springer.com/article/10.1007/s13735-021-00223-4)

#### Attacks

- FGSM [Explaining and Harnessing Adversarial Examples](http://arxiv.org/abs/1412.6572)
- PGD [Towards Deep Learning Models Resistant to Adversarial Attacks](http://arxiv.org/abs/1706.06083)
- [DeepFool: a simple and accurate method to fool deep neural networks](http://arxiv.org/abs/1511.04599) 
- Carline and Wagner's attacks[Towards Evaluating the Robustness of Neural Networks](http://arxiv.org/abs/1608.04644)

### Example Images
Conventional Image Rectification Framework using Ensemble CNN
![Ensemble](https://github.com/Caryox/adversial-robustness/blob/fc70b735438bafb5d275c1ca33cc52e5209739bc/data/Ensemble%20Conventional%20Rectification%20V4_Updated.jpg)
Image processing
![Image processing](https://github.com/Caryox/adversial-robustness/blob/fc70b735438bafb5d275c1ca33cc52e5209739bc/data/example_image_processing.jpg)

Adversarial Detection
![grafik](https://user-images.githubusercontent.com/56730144/187693742-67b5af57-58a6-4785-8814-6b1ce7fc5e34.png)
Try to calculate best threshold t for seperate adversarials and legitimate samples by measuring the maximum L1-Distance of the SoftMax-Layer between the baseline CNN ensemble and the feature squeezed CNN ensembles


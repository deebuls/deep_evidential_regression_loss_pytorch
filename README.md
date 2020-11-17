---

<div align="center">    
 
# Deep Evidential Regression Loss Function

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/1910.02600)
[![Conference](http://img.shields.io/badge/ICLR-2020-4b44ce.svg)](https://openreview.net/forum?id=S1eSoeSYwr)

<!--  
Conference   
-->   
</div>
 
## Description   
The paper "Deep Evidential Uncertainty/Regression" was submitted to ICLR where
it was rejected[1]. The idea is inline with light of Sensoy et al.[2] and Malinin & Gales[3].
It was rejected because of lack of experiments and similar ideas with Malinin thesis.
The goal is to implement the loss function and validate the results.


## Installation

### Typical Install
```
pip install git+https://github.com/deebuls/deep_evidential_regression_loss_pytorch
```

### Development
```
git clone https://github.com/deebuls/deep_evidential_regression_loss_pytorch
cd deep_evidential_regression_loss_pytorch
pip install -e .[dev]
```

Tests can then be run from the root of the project using:
```
nosetests
```

## Usage

To use this code `EvidentialLossSumOfSquares` and create loss function. `loss.py`
implements the evidential loss function.

Check examples for detailed usage example

## ToDo

1. Different variation of the loss (NLL, with log(alpha, beta, lambda))
2. When output is image as case of VAE
3. Examples
4. Test cases


## Abstract
Deterministic neural networks (NNs) are increasingly being deployed in safety
critical domains, where calibrated, robust and efficient measures of
uncertainty are crucial. While it is possible to train regression networks to
output the parameters of a probability distribution by maximizing a Gaussian
likelihood function, the resulting model remains oblivious to the underlying
confidence of its predictions. In this paper, we propose a novel method for
training deterministic NNs to not only estimate the desired target but also the
associated evidence in support of that target. We accomplish this by  placing
evidential priors over our original Gaussian likelihood function and training
our NN to infer the hyperparameters of our evidential distribution. We impose
priors during training such that the model is penalized when its predicted
evidence is not aligned with the correct output. Thus the model estimates not
only the probabilistic mean and variance of our target but also the underlying
uncertainty associated with each of those parameters. We observe that our
evidential regression method learns well-calibrated measures of uncertainty on
various benchmarks, scales to complex computer vision tasks, and is robust to
adversarial input perturbations.

## References
* [1] https://openreview.net/forum?id=S1eSoeSYwr&noteId=78WcDK50Bi
* [2] M. Sensoy, et al. "Evidential deep learning to quantify classification uncertainty." NeurIPS. 2018.
* [3] A. Malinin, et al. Predictive uncertainty estimation via prior networks. NeurIPS 2018.

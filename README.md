# HarsanyiNet
This repository contains the Python implementation for HarsanyiNet, "HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation", ICML 2023.

HarsanyiNet is an interpretable network architecture, which makes inferences on the input sample and simultaneously computes the exact Shapley values of the input variables in a single forward propagation (see [papers]() for details and citations).

## Install
HarsanyiNet can be installed in the Python 3 environment:

`
pip install git+https://github.com/csluchen/harsanyinet
`



## How to use easily
### HarsanyiNet-CNN
To train CIFAR-10 dataset easily, you can use the following code:

`
python train.py
`
Or you can directly access the trained model in path A.

To train MNIST dataset easily, you can use the following code:

`python train.py --dataset='MNIST'`

Or you can directly access the trained model in path B.

Now you can use the HarsanyiNet to compute the Shapley values in a single forward propagation, you can use:

`
python shapley.py --model_path='layers10_channels256_beta1000_gamma1'
`





### HarsanyiNet-MLP



## More details
If you want to train a tranditional DNN with comparable size, you can use the following code:

`
python train.py --comparable_DNN
`

If you want to compare the Shapley values calculated by the HarsanyiNet with 




## Sample notebooks




## Citations

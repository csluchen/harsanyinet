# HarsanyiNet
This repository contains the Python implementation for HarsanyiNet, "HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation", ICML 2023.

HarsanyiNet is an interpretable network architecture, which makes inferences on the input sample and simultaneously computes the exact Shapley values of the input variables in a single forward propagation (see [papers]() for details and citations).

## Install
HarsanyiNet can be installed in the Python 3 environment:

`
pip install git+https://github.com/csluchen/harsanyinet
`

In addition, the `torchtoolbox` package needs to be installed:

`
pip install torchtoolbox
`



## How to use 
### HarsanyiNet-CNN
#### Get HarsanyiNet model
To train CIFAR-10 dataset easily, you can use the following code:

`
python train.py
`

Or you can directly access the pre-trained HarsanyiNet in path `A`.

To train MNIST dataset easily, you can use the following code:

`python train.py --dataset='MNIST'`


Or you can directly access the pre-trained HarsanyiNet in path `B`.

#### Compute Shapley values by HarsanyiNet
Now you can use the trained HarsanyiNet to compute the Shapley values in a single forward propagation, you can use the following code like:

`
python shapley.py 
`





### HarsanyiNet-MLP

### Datasets

We provide implementation on three different tabular datasets from UCI repository, including

- [Census income](https://archive.ics.uci.edu/ml/datasets/census+income)
- [Yeast](https://archive.ics.uci.edu/ml/datasets/Yeast) 
- [Commercial (TV News)](http://archive.ics.uci.edu/ml/datasets/tv+news+channel+commercial+detection+dataset) 

### Getting Started

To get started, you can run `python utils/tabular/data_preprocess.py` to download and preprocess the data. The preprocessed data will be stored as  an`np.ndarry` in `data/{DATASET}/`. Alternatively, you can directly use `data/data.py` to load the dataloader directly, we have already incorporate this step. 

To train the model, use the following code:

- Census dataset `python train_tabular.py`
- Yeast dataset `python train_tabular.py --dataset Yeast --n_attributes 8`
- Commercial (TV News) dataset `python train_tabular.py --dataset Commercial --n_attributes 10`

We also provide the trained models under `model_pth/` (please note that the Yeast and Commercial dataset don't have official data splits. We randomly split the whole dataset into 80% training data and 20% testing data. Therefore, the results may vary.)



To compute Shapley values using HarsanyiNet in a single forward propagation, use the following code:

`python shapley_tabular.py --harsanyinet --model_path model_pth/Census.pth`



## More details
#### Compare Shapley values computed by HarsanyiNet and Other methods

If you want to compute the root mean squared error (RMSE) between the Shapley values computed by HarsanyiNet and that computed by sampling method, you can use the following code:

`
python shapley.py --sampling=True --runs=2000
`

Or you can directly access the sampling result in path `C`.



For image dataset, if you want to compute the root mean squared error (RMSE) between the Shapley values computed by HarsanyiNet and that computed by sampling method, you can use the following code:

`
python shapley.py ...
`


## Sample notebooks

We provide a Jupyter notebook for the Census dataset for calculating Shapley values via HarsanyiNet under `notebooks/Census.ipynb`


## Citations

# HarsanyiNet
This repository contains the Python implementation for HarsanyiNet, "HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation", ICML 2023.

HarsanyiNet is an interpretable network architecture, which makes inferences on the input sample and simultaneously computes the exact Shapley values of the input variables in a single forward propagation (see [papers](https://arxiv.org/abs/2304.01811) for details and citations).

## Install
HarsanyiNet can be installed in the Python 3 environment:

```
pip3 install git+https://github.com/csluchen/harsanyinet
```

The `torchtoolbox` package also needs to be installed:

```
pip3 install torchtoolbox
```

Besides, if you want to compare HarsanyiNet with other methods, please install the following dependencies.

```
pip3 install -r requirements.txt
```


## How to use 
### HarsanyiNet-CNN
To train the model, you can use codes like the following:

- CIFAR-10 dataset 

```
python train.py
```

- MNIST dataset 

```
python train.py --dataset='MNIST' --num_layers=4 --channels=32 --beta=100 --gamma=0.05
```

or directly access the pre-trained HarsanyiNet in path ```./output/{DATASET}/.../model_pths/{DATASET}.pth```.

To compute Shapley values using HarsanyiNet in a single forward propagation, use codes like the following:

```
python shapley.py --model_path='model_pths/CIFAR10.pth' --num_layers=10 --channels=256 --beta=1000 --gamma=1 
```





### HarsanyiNet-MLP

### Datasets

We provide implementation on three different tabular datasets from UCI repository, including

- [Census income](https://archive.ics.uci.edu/ml/datasets/census+income)
- [Yeast](https://archive.ics.uci.edu/ml/datasets/Yeast) 
- [Commercial (TV News)](http://archive.ics.uci.edu/ml/datasets/tv+news+channel+commercial+detection+dataset) 

### Getting Started

To get started, you can run ```python utils/tabular/data_preprocess.py``` to download and preprocess the data. The preprocessed data will be stored as  an`np.ndarry` in `data/{DATASET}/`. Alternatively, you can directly use ```utils/data.py``` to load the dataloader directly, we have already incorporate this step. 

To train the model, use the following code:

- Census dataset

```
python train_tabular.py
```

- Yeast dataset 

```
python train_tabular.py --dataset Yeast --n_attributes 8
```

- Commercial (TV News) dataset 

```
python train_tabular.py --dataset Commercial --n_attributes 10
```

(please note that the Yeast and Commercial dataset don't have official data splits. We randomly split the whole dataset into 80% training data and 20% testing data. Therefore, the results may vary.)



To compute Shapley values using HarsanyiNet in a single forward propagation, use the following code:

- Census
```
python shapley_tabular.py --harsanyinet 
```
- Yeast
```
   python shapley_tabular.py --dataset Yeast --n_attributes 8 --harsanyinet
```
- Commercial
```
   python shapley_tabular.py --dataset Commercial --n_attributes 10 --harsanyinet
```



## More details
### Comparing Shapley values computed by HarsanyiNet and other methods

To compute the root mean squared error (RMSE) between the Shapley values computed by HarsanyiNet and sampling method, use the following code:

```
python shapley.py --sampling=True --runs=20000
```

*Note:* the larger the number of iterations (runs) of the sampling method, the more accurate the sampling method is and the longer it takes for the code to run.

To compute the RMSE between the Shapley values computed by HarsanyiNet and ground-truth Shapley values, use the following code:

```
python shapley.py --ground_truth=True
```


## Sample notebooks

For image dataset, we provide a Jupyter notebook for the `CIFAR-10` and `MNIST` dataset for calculating Shapley values via HarsanyiNet under ```notebooks/CIFAR-10.ipynb``` and ```notebooks/MNIST.ipynb```, respectively.

For tabular dataset, we provide a Jupyter notebook for the `Census` dataset for calculating Shapley values via HarsanyiNet under ```notebooks/Census.ipynb```


## Citations
```
@InProceedings{chen23,
  title = {HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation},
  author = {Lu, Chen and Siyu, Lou and Keyan, Zhang and Jin, Huang and Quanshi, Zhang},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  year = {2023}
}
```

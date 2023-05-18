import os
import numpy as np
import torch
from utils.tabular.shapreg import removal, games, shapley

from utils.attribute import HarsanyiMLPAttribute
from utils.tabular.shap_util import brute_force_shapley, ShapSampling, ShapKernel, permutation_sample_parallel
from utils.tabular.plot import plot_convergence

def get_sample(test_loader, index=0, batch=0, batch_size=1,device='cuda:0'):
    for i, (x_te, y_te) in enumerate(test_loader):
        x_te = x_te.to(device)
        y_te = y_te.to(device)
        if i == batch:
            if batch_size == 1:
                return x_te[index].unsqueeze(0), y_te[index]
            else:
                return x_te, y_te

def check_shape(shapley):
    if len(shapley.shape)>1:
        shapley = shapley.reshape(-1)
    return shapley

def HarsanyiNetShapley(model, x_te, label):
    '''function to estimate Shapley values using HarsanyiNet'''
    device = model.device
    calculator = HarsanyiMLPAttribute(model=model, device=device)
    harsanyi = calculator.attribute(model=model, x_te=x_te, target_label=label)
    Harsanyi_Shapley = calculator.get_shapley(harsanyi=harsanyi)

    return Harsanyi_Shapley

def BruteForceShapley(model, x_te, label):
    '''function to calculate the ground truth Shapley values from definition'''
    # the reference value is set to 0 for each input variable
    device = model.device
    reference = torch.zeros(x_te.shape[-1]).to(device) 
    shapley_bf = brute_force_shapley(model, x_te, reference, label)
    shapley_bf = shapley_bf.detach().cpu().numpy()
    return check_shape(shapley_bf)

def SamplingShapley(model, x_te, label, runs):
    '''function to estimate the Shapley values via sampling
    :param runs: number of samplings
    '''
    shapley = ShapSampling(model, x_te, label, n_samples=runs)
    Sampling_Shapley = shapley.squeeze().squeeze().detach().cpu().numpy()
    
    return check_shape(Sampling_Shapley)

def PermutationSamplingShapley(model, x_te, label, runs):
    '''function to estimate the Shapley values via antithetical sampling
    '''
    device = model.device
    reference = torch.zeros(x_te.shape[-1]).to(device)
    shapley =permutation_sample_parallel(model, x_te, reference, device, label,batch_size=runs, antithetical=True)
    Permutation_Shapley = shapley.detach().cpu().numpy()
    return check_shape(Permutation_Shapley)


def KernelShapley(model, x_te, label, runs ):
    '''function to estimate the Shapley values via KernelSHAP'''
    reference = torch.zeros((128,x_te.shape[-1]))
    marginal_extension = removal.MarginalExtension(reference, model)
    game = games.PredictionGame(marginal_extension, x_te)
    kernel = shapley.ShapleyRegression(game, paired_sampling=False,
                                       detect_convergence = False,
                                       n_samples = runs,
                                       batch_size=32,
                                       bar=False)
    KernelShapley = kernel.values[:,label]
    return check_shape(KernelShapley)

def KernelPairShapley(model, x_te, label, runs):
    '''function to estimate Shapley values via KernelShap pair'''
    reference = torch.zeros((128, x_te.shape[-1]))
    marginal_extension = removal.MarginalExtension(reference, model)
    game = games.PredictionGame(marginal_extension, x_te)
    kernel = shapley.ShapleyRegression(game, paired_sampling=True,
                                       detect_convergence = False,
                                       n_samples = runs//2,
                                       batch_size=64,
                                       bar=False)
    KernelPairShapley = kernel.values[:,label]
    return check_shape(KernelPairShapley)


def get_RMSE(method1, method2, str='', n_players=None):
    '''function to compute the root mean square error of the estimated Shapleyvalues'''
    gt = method1.reshape(-1)
    value = method2.reshape(-1)
    if n_players is None:
        dim = gt.shape[0]
    else:
        dim = n_players
        print("num of players:", dim)

    loss_abs = np.abs(value - gt)
    RMSE = np.sqrt((loss_abs**2).sum() / dim)
    print(f"RMSE of {str}:", RMSE)
    return RMSE

def plot_shapleys(args, model, test_loader, device,save_dir):
    from tqdm import tqdm
    bfs = []
    harsanyis = []
    samplings = []
    permutations = [] 
    kernels = []
    pairs = []
    
    for index in tqdm(range(args.num_samples)):

        # get data
        x_te, y_te = get_sample(test_loader, index=index ,device=device)
        label = int(y_te)
        x_te = x_te.double()
        model = model.double()
        baseline = torch.zeros_like(x_te).to(device)
        
        # Get the ground truth Shapley value
        Shapley_bf = BruteForceShapley(model, x_te, label)
        bfs.append(Shapley_bf)

        Harsanyi_Shap = HarsanyiNetShapley(model, x_te, label)
        harsanyis.append(Harsanyi_Shap)    
        sampling_single, permutation_single, kernel_single, pair_single = [], [], [], [] 
        for runs in [1,10,32, 50, 100, 200, 500, 1000, 1500, 2000, 5000]: 
            Sampling_Shapley = SamplingShapley(model, x_te, label, runs)
            sampling_single.append(Sampling_Shapley)

            Permutation_Shapley = PermutationSamplingShapley(model,x_te, label, runs)
            permutation_single.append(Permutation_Shapley)

            Kernel_Shap = KernelShapley(model, x_te, label, runs)
            kernel_single.append(Kernel_Shap)
        
        for runs in [50, 100, 200, 500, 1000, 1500, 2000, 5000]:   
            Kernel_Shap_Pair = KernelPairShapley(model, x_te, label, runs)
            pair_single.append(Kernel_Shap_Pair)
         
        samplings.append(sampling_single)    
        permutations.append(permutation_single)
        kernels.append(kernel_single)
        pairs.append(pair_single)
        
    shapley_value_bf = np.asarray(bfs)
    shapley_value_harsanyi = np.asarray(harsanyis)
    shapley_sampling = np.asarray(samplings)
    shapley_sampling_permutation = np.asarray(permutations)
    shapley_kernel = np.asarray(kernels)
    shapley_kernel_pair = np.asarray(pairs)
    
    np.save(os.path.join(save_dir,'trueShapley.npy'), shapley_value_bf)
    np.save(os.path.join(save_dir,'HarsanyiShapley.npy'), shapley_value_harsanyi)    
    np.save(os.path.join(save_dir,'SamplingShapley.npy'),shapley_sampling)
    np.save(os.path.join(save_dir,'SamplingAntitheticalShapley.npy'), shapley_sampling_permutation)
    np.save(os.path.join(save_dir,'KernelShapley.npy'), shapley_kernel)
    np.save(os.path.join(save_dir,'KernelPairShapley.npy'), shapley_kernel_pair)

    attr_dic= {'HarsanyiShapley':shapley_value_harsanyi,
                'SamplingShapley':shapley_sampling,
                'KernelShapley':shapley_kernel,
                'KernelPairShapley':shapley_kernel_pair,
                'AntitheticalShapley':shapley_sampling_permutation}
    plot_convergence(shapley_value_bf, attr_dic, save_dir)

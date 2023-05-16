'''
The code is modified based on the repository
https://github.com/guanchuwang/SHEAR/blob/main/shap_utils.py

'''
from scipy.special import comb, perm
import torch
import numpy as np
from tqdm import tqdm

from captum.attr import ShapleyValueSampling, KernelShap

def binary(x, bits):
    mask = 2 ** torch.arange(bits)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

@torch.no_grad()
def sub_brute_force_shapley(f, x, reference, mask, feature_index, M, gt):

    set_0 = torch.cat((mask, torch.zeros((mask.shape[0], 1)).byte()), dim=1)
    set_1 = torch.cat((mask, torch.ones((mask.shape[0],1)).byte()), dim=1)

    device = x.device
    set_0[:,[feature_index, -1]] = set_0[:, [-1, feature_index]]
    set_1[:, [feature_index,-1]] = set_1[:, [-1, feature_index]]
    S = set_0.sum(dim=1)
    weights = 1. / torch.from_numpy(comb(M-1, S)).type(torch.double).to(device)
    
    set_0, set_1 = set_0.to(device), set_1.to(device)
    reference = reference.to(device)
    f_set_0_input = (set_0 * x + ( 1 - set_0) * reference.unsqueeze(dim=0))
    f_set_1_input = (set_1 * x + (1 - set_1) * reference.unsqueeze(dim=0))
    f_set_0 = f(f_set_0_input.to(device))
    f_set_1 = f(f_set_1_input.to(device))
    shapley_value = 1./M*weights.unsqueeze(dim=0).mm(f_set_1- f_set_0)
    return shapley_value.squeeze()[gt]

@torch.no_grad()
def sub_brute_force_shapley_sep(f, x, reference, mask_dec, feature_index, M, gt):

    num_chuncks = int(2**(M-1) / 2**20)
    f_set_delta = []
    for k in tqdm(range(num_chuncks)):
        mask = binary(mask_dec[k*2**20:(k+1)*2**20],M-1)

        set_0 = torch.cat((mask, torch.zeros((mask.shape[0], 1)).byte()), dim=1)
        set_1 = torch.cat((mask, torch.ones((mask.shape[0],1)).byte()), dim=1)

        set_0[:,[feature_index, -1]] = set_0[:, [-1, feature_index]]
        set_1[:, [feature_index,-1]] = set_1[:, [-1, feature_index]]
        S = set_0.sum(dim=1)
        device = x.device
        reference = reference.to(device)
        set_0, set_1 = set_0.to(device), set_1.to(device)
        f_set_0_input = (set_0 * x + ( 1 - set_0) * reference.unsqueeze(dim=0))
        f_set_1_input = (set_1 * x + (1 - set_1) * reference.unsqueeze(dim=0))
        f_set_0 = f(f_set_0_input.to(device))
        f_set_1 = f(f_set_1_input.to(device))
        f_set_delta.append(f_set_1-f_set_0)
    f_set_delta = torch.cat(f_set_delta)
    weights = 1. / torch.from_numpy(comb(M-1, S)).type(torch.double)
    shapley_value = 1./M*weights.unsqueeze(dim=0).mm(f_set_delta)
    return shapley_value.squeeze()[gt]

@torch.no_grad()
def brute_force_shapley(f, x, reference, gt, shap_index=None, batch_size=None):
    f = f.double()
    x = x.double()
    if len(x.shape)>2:
        x = x.squeeze(1)
    M = x.shape[-1]
    shap_index = torch.arange(M) if shap_index is None else shap_index
    mask_dec = torch.arange(0, 2 ** (M-1))
    shapley_value = torch.zeros((x.shape[0], len(shap_index))).double()
    if len(mask_dec) > 2**20:
        for idx, feature_index in enumerate(shap_index):
            shapley_value[:,idx] = sub_brute_force_shapley_sep(f,x,reference,mask_dec,feature_index,M,gt) 
    else:
        mask = binary(mask_dec, M-1)
        for idx, feature_index in enumerate(shap_index):
            shapley_value[:, idx] = sub_brute_force_shapley(f, x, reference, mask, feature_index, M, gt)
    return shapley_value

def ShapSampling(model, x,y, n_samples):
    shap_sampling = ShapleyValueSampling(model)
    attr_sampling = shap_sampling.attribute(x, target=y, n_samples=n_samples, perturbations_per_eval=1000)
    return attr_sampling

def ShapKernel(model, x, y, n_samples):
    ks = KernelShap(model)
    attr_kernel = ks.attribute(x, target = y, n_samples=n_samples, perturbations_per_eval=1000)
    return attr_kernel

def f_mask(f, x, reference, S,device):
    S = S.to(device)
    x_mask = S * x + (1 - S) * reference.unsqueeze(dim=0)
    return f(x_mask)

@torch.no_grad()
def permutation_sample_parallel(f, x, reference,device, target, batch_size=16, antithetical=True):

    M = x.shape[-1]
    queue = torch.arange(M).unsqueeze(dim=0).repeat((batch_size, 1))

    for idx in range(batch_size):
        if antithetical and idx > batch_size//2:
            queue[idx] = torch.flip(queue[batch_size-1-idx], dims=(0,))
        else:
            queue[idx] = queue[idx, torch.randperm(M)]

    arange = torch.arange(batch_size)
    deltas = torch.zeros_like(queue).type(torch.double)
    deltas = deltas.to(device)

    S = torch.zeros_like(queue).type(torch.long)

    S_buf = []
    for index in range(M):
        S[arange, queue[:, index]] = 1
        S_buf.append(S.clone())

    for index in range(M):
        S = S_buf[index]
        if index == 0:
            deltas[arange, queue[:, index]] = (f_mask(f, x, reference, S,device) - f(reference.unsqueeze(dim=0).double()).repeat((batch_size, 1)))[:,target]
        else:
            S_ = S_buf[index-1]
            deltas[arange, queue[:, index]] = (f_mask(f, x, reference, S,device) - f_mask(f, x, reference, S_,device))[:,target]


    return deltas.mean(dim=0).unsqueeze(dim=0)


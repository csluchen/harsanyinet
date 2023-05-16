import torch
import torch.nn.functional as F
import numpy as np

'''
Implementation of "Polynomial calculation of the Shapley value based on sampling"
From https://www.sciencedirect.com/science/article/pii/S0305054808000804
Modified from code: https://github.com/marcoancona/DASP/blob/master/utils/shapley_sampling.py
'''

def shapley_sampling(model, image, target_label, device, grid_width=1, runs=200):
    """
    :param target_label: output dimension (before softmax) which have to be attributed
    :param grid_width: size of a super pixel / pixel
    :param runs: number of network inferences
    """

    shape = list(image.shape)
    width = image.shape[2]
    sqrt_grid_num = int(np.ceil(width / grid_width))
    n_features = sqrt_grid_num ** 2      # number of super pixels / pixels
    shap_value = np.zeros((shape[0], n_features))


    for r in range(runs):
        p = np.random.permutation(n_features)
        x = image.clone()
        mask_grid = torch.zeros((image.shape[0], image.shape[1], sqrt_grid_num, sqrt_grid_num)).to(device)

        y0 = None
        # for feature i in permutation p
        for i in p:
            if y0 is None:  # get v( Pre(p) ) = v(\empty)
                mask = F.interpolate(mask_grid.clone(), size=[grid_width * sqrt_grid_num, grid_width * sqrt_grid_num],mode="nearest").float()
                y0 = model(mask * x)
                
            # add feature i
            mask_grid = mask_grid.reshape(image.shape[0], image.shape[1], -1)
            mask_grid[:, :, i] = 1
            mask_grid = mask_grid.reshape(image.shape[0], image.shape[1], sqrt_grid_num, sqrt_grid_num)
            mask = F.interpolate(mask_grid.clone(), size=[grid_width * sqrt_grid_num, grid_width * sqrt_grid_num], mode="nearest").float()

            # get v( Pre(p) U {i} )
            y = model(mask * x)

            # get v( Pre(p) U {i} ) - v( Pre(p) )
            pred_diff = y[:, target_label] - y0[:, target_label]

            # update shapley value of feature i
            shap_value[0, i] += pred_diff

            # update v( Pre(p) )
            y0 = y

    shap_value = shap_value.copy() / runs

    return shap_value

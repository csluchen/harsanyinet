import torch
from scipy.special import comb

'''
Implementation of Ground-truth Shapley values of selected input variables in a sample
Modified from code: https://github.com/guanchuwang/SHEAR/blob/10baebc052845b891409364f05e1c78a75e614ff/shap_utils.py#L93
'''

def binary(x, bits):
    mask = 2 ** torch.arange(bits)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

@torch.no_grad()
def sub_brute_force_shapley(f, device, x, gt_label, reference, players, mask, feature_index, M, dataset):
    set0 = torch.cat((mask, torch.zeros((mask.shape[0], 1)).byte()), dim=1)
    set1 = torch.cat((mask, torch.ones((mask.shape[0], 1)).byte()), dim=1)

    set0[:, [feature_index, -1]] = set0[:, [-1, feature_index]]  # S
    set1[:, [feature_index, -1]] = set1[:, [-1, feature_index]]  # S U {i}
    S = set0.sum(dim=1)  # |S|

    weights = 1. / torch.from_numpy(comb(M-1, S)).type(torch.float).to(device)

    index = torch.tensor([[i // x.shape[2], i % x.shape[2]] for i in players]).clone()
    input_set0, input_set1 = [], []
    for s0, s1 in zip(set0, set1):

        # get input x_{S}
        list = [n for i, n in enumerate(index) if i in torch.where(s0>0)[0].numpy().tolist()]
        if list != []:
            idx0 = torch.stack(list)
            masked_x = torch.zeros_like(x)
            masked_x[0, :, idx0[:, 0], idx0[:, 1]] = x[0, :, idx0[:, 0], idx0[:, 1]]
        input_s0 = reference + masked_x if list != [] else reference

        # get input x_{S U {i}}
        idx1 = torch.stack([n for i, n in enumerate(index) if i in torch.where(s1 > 0)[0].numpy().tolist()])
        masked_x = torch.zeros_like(x)
        masked_x[0, :, idx1[:, 0], idx1[:, 1]] = x[0, :, idx1[:, 0], idx1[:, 1]]
        input_s1 = reference + masked_x

        if input_set0 == []:
            input_set0 = input_s0
            input_set1 = input_s1
        else:
            input_set0 = torch.cat((input_set0, input_s0), axis=0)
            input_set1 = torch.cat((input_set1, input_s1), axis=0)

    if dataset == 'MNIST':  # If sufficient memory
        f_set0 = f(input_set0)[:, gt_label]
        f_set1 = f(input_set1)[:, gt_label]

    elif dataset == 'CIFAR10':  # If insufficient memory
        batch_size = int(input_set0.shape[0] / 16)
        f_set0, f_set1 = torch.zeros(1), torch.zeros(1)
        for i in range(0, input_set0.shape[0], batch_size):
            input_set0_batch = input_set0[i:i+batch_size, :]
            input_set1_batch = input_set1[i:i+batch_size, :]
            f_set0_batch = f(input_set0_batch)[:, gt_label]
            f_set1_batch = f(input_set1_batch)[:, gt_label]
            if f_set0.shape[0] == 1:
                f_set0 = f_set0_batch
                f_set1 = f_set1_batch
            else:
                f_set0 = torch.cat((f_set0, f_set0_batch), dim=0)
                f_set1 = torch.cat((f_set1, f_set1_batch), dim=0)

    shapley_value = 1./M * (weights *(f_set1 - f_set0)).sum()

    return shapley_value


@torch.no_grad()
def brute_force_shapley_mask(model, device, image, target_label, n_players, baseline, players, dataset='MNIST'):
    M = n_players
    shap_index = torch.arange(M)
    mask_dec = torch.arange(0, 2 ** (M-1))
    mask = binary(mask_dec, M - 1)

    shapley_value = torch.zeros((image.shape[0], len(shap_index)))
    for idx, feature_idx in enumerate(shap_index):
        shapley_value[:, idx] = sub_brute_force_shapley(model, device, image, target_label, baseline, players,
                                                        mask, feature_idx, M, dataset).squeeze(dim=0)

    shapley_value_mask = torch.zeros((image.shape[2], image.shape[3]))
    index = torch.tensor([[i // image.shape[2], i % image.shape[2]] for i in players])
    shapley_value_mask[index[:, 0], index[:, 1]] = shapley_value

    return shapley_value_mask.detach().cpu().numpy()

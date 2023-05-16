import os
import argparse
import torch
import numpy as np

from model.HarsanyiNet import HarsanyiNet
from utils.attribute import HarsanyiNetAttribute

from utils.image.shapley_sampling import shapley_sampling
from utils.image.attribute_mask import HarsanyiNetAttributeMask
from utils.image.groundtruth_mask import brute_force_shapley_mask

from utils.data import get_data_loader
from utils.plot import plot_shapley
from utils.seed import setup_seed


parser = argparse.ArgumentParser(description='HarsanyiNet to compute Shapley values')
parser.add_argument('--dataset', type=str, default='CIFAR10', help=" 'CIFAR10', 'MNIST' can be chosen")
parser.add_argument('--batch_size', type=int, default=50)

parser.add_argument('--seed', type=str, default=0)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save_path', type=str, default='./output')
parser.add_argument('--model_path', type=str, default='model_pths/CIFAR10.pth')

# parameters for HarsanyiNet
parser.add_argument('--gamma', type=float, default=1, help="a postive value in the tanh function. \
                                                      The larger the gamma, the output of tanh function closer to 1.")
parser.add_argument('--beta', type=int, default=1000, help="a postive value for back propagation. In back propagation, \
                                                      the sigmoid function is used to approximate the indicator function.")
parser.add_argument('--num_layers', type=int, default=10, help="number of layers")
parser.add_argument('--channels', type=int, default=512, help="number of channels")
parser.add_argument('--comparable_DNN', type=bool, default=False, help="whether to use a tranditional DNN with comparable size, \
                                                                        False - HarsanyiNet, True - Traditional DNN")

# parameters for attribution
parser.add_argument('--harsanyinet', type=bool, default=True, help="whether to use HarsanyiNet to compute Shapley values.")
parser.add_argument('--input_type', type=str, default='z0', help="For the image dataset, attribution to 'z0'")
parser.add_argument('--sampling', type=bool, default=False, help="whether to use Sampling to compute Shapley values.")
parser.add_argument('--runs', type=int, default=2000, help="number of iterations if using Sampling method")
parser.add_argument('--ground_truth', type=bool, default=False, help="whether to compare Shapley values by HarsanyiNet  \
                                                               with ground-truth Shapley values.")

parser.add_argument('--test_acc', type=bool, default=False, help="whether to test the classification accuracy of the model")

args = parser.parse_args()


# compute Shapley values of all input variables in a sample
def HarsanyiNetShapley(model, x_te, label):
    calculator = HarsanyiNetAttribute(model=model, device=device)
    harsanyi = calculator.attribute(model=model, image=x_te, target_label=label)
    Harsanyi_Shapley = calculator.get_shapley(harsanyi=harsanyi)

    return Harsanyi_Shapley

def SamplingShapley(model, x_te, label, runs=args.runs):
    shapley = shapley_sampling(model=model, image=x_te, target_label=label, device=device, runs=runs)
    Sampling_Shapley = shapley.reshape(x_te.shape[-1], x_te.shape[-1])

    return Sampling_Shapley


# compute Shapley values of selected input variables in a sample
def HarsanyiNetShapley_mask(model, x_te, label, n_players, baseline, players):
    calculator = HarsanyiNetAttributeMask(model=model, device=device, n_players=n_players, baseline=baseline, players=players)
    harsanyi = calculator.attribute(model=model, image=x_te, target_label=label)
    Harsanyi_Shapley = calculator.get_shapley(harsanyi=harsanyi)

    return Harsanyi_Shapley

def GroundTruthShapley_mask(model, x_te, label, n_players, baseline, players, dataset):
    Groundtruth_Shapley = brute_force_shapley_mask(model=model, device=device, image=x_te, target_label=label, n_players=n_players,
                                              baseline=baseline, players=players, dataset=dataset)

    return Groundtruth_Shapley


def get_sample(test_loader, index=0, batch=0, batch_size=1):
    for i, (x_te, y_te) in enumerate(test_loader):
        x_te = x_te.to(device)
        y_te = y_te.to(device)
        if i == batch:
            if batch_size == 1:
                return x_te[index].unsqueeze(0), y_te[index]
            else:
                return x_te, y_te


def get_RMSE(method1, method2, str, n_players=None):
    gt = method1.reshape(-1)
    value = method2.reshape(-1)

    if n_players is None:
        dim = gt.shape[0]
    else:
        dim = n_players
        print("num of players:", dim)

    loss_abs = np.abs(value - gt)
    RMSE = np.sqrt((loss_abs**2).sum() / dim)
    print(f"RMSE of {str}: {RMSE.item()}")
    return RMSE


def test(model, device, test_loader):
    test_correct_value = 0
    test_total_num = 0

    for i, (x_te, y_te) in enumerate(test_loader):
        x_te = x_te.to(device)
        y_te = y_te.to(device)

        with torch.no_grad():
            y_pred = model(x_te)
            test_correct_value += (y_pred.max(1)[1] == y_te).sum().item()
            test_total_num += x_te.size(0)

    avg_te_acc = test_correct_value / test_total_num
    print(f"test_acc: {avg_te_acc:.4f}\n")


def init_path(args):
    path_dir = f"layers{args.num_layers}_channels{args.channels}_beta{args.beta}_gamma{args.gamma}"
    args.model_path = os.path.join(args.save_path, args.dataset, path_dir, args.model_path)
    if not os.path.exists(args.model_path):
        print(f"Path {args.model_path} does not exist.")
        exit(0)
    print("Load model:", args.model_path)


def get_path(args, str, index=0):
    if str == 'harsanyinet':
        filename = f"harsanyinet_{index}"
        shapley_path = os.path.join(args.save_path, "HarsanyiNet", filename)
        if not os.path.exists(shapley_path):
            if not os.path.exists(os.path.join(args.save_path, "HarsanyiNet")):
                os.makedirs(os.path.join(args.save_path, "HarsanyiNet"))

    elif str == 'sampling':
        filename = f"sampling_{args.runs}_{index}"
        shapley_path = os.path.join(args.save_path, "Sampling", filename)
        if not os.path.exists(shapley_path):
            if not os.path.exists(os.path.join(args.save_path, "Sampling")):
                os.makedirs(os.path.join(args.save_path, "Sampling"))

    return shapley_path



if __name__ == '__main__':
    init_path(args)
    setup_seed(args.seed)

    train_loader, test_loader, num_classes = get_data_loader(args.dataset, args.batch_size)
    device = args.device if torch.cuda.is_available() else 'cpu'

    in_channels = 3
    if args.dataset == 'MNIST':
        in_channels = 1
    model = HarsanyiNet(num_layers=args.num_layers, channel_extend=args.channels,
                        beta=args.beta, gamma=args.gamma,
                        num_classes=num_classes, device=device, in_channels=in_channels,
                        comparable_DNN=args.comparable_DNN).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    if args.test_acc:
        test(model, device, test_loader)

    if args.comparable_DNN:
        args.harsanyinet = False
        print("Compute Shapley values on a traditional DNN. Only the sampling method can be used.")

    RMSE = 0
    for index in range(args.batch_size):

        # get image
        x_te, y_te = get_sample(test_loader, index=index)
        label = int(y_te)
        model = model.double()  # double float to ensure accurate attribution
        x_te = x_te.double()

        # get z0
        if args.input_type == 'z0':
            z0 = model._get_z0(x_te)
        z0 = z0.double()
        baseline = torch.zeros_like(z0).to(device)


        # HarsanyiNet to compute Shapley values
        if args.harsanyinet:
            harsanyi_path = get_path(args, 'harsanyinet', index=index)
            Harsanyi_Shap = HarsanyiNetShapley(model, z0, label)
            plot_shapley(Harsanyi_Shap, path=harsanyi_path, str=f'{index} with label {label} (HarsanyiNet)')


        # Sampling method to compute Shapley values
        if args.sampling:
            sampling_path = get_path(args, 'sampling', index=index)
            if not os.path.exists(sampling_path+'.npy'):
                Sampling_Shap = SamplingShapley(model, z0, label, runs=args.runs)
                np.save(sampling_path+'.npy', Sampling_Shap)
            else:
                Sampling_Shap = np.load(sampling_path+'.npy')
            vmax = max(Harsanyi_Shap.max(), -Harsanyi_Shap.min())
            plot_shapley(Sampling_Shap, path=sampling_path, str=f'{index} (Sampling {args.runs})', vmax=vmax)


        # compare HarsanyiNet and Sampling method to compute Shapley values
        if args.harsanyinet and args.sampling:
            get_RMSE(Harsanyi_Shap, Sampling_Shap, f"HarsanyiNet (1) and Sampling method ({args.runs})")


        # compare Shapley values computed by HarsanyiNet and ground-truth Shapley values
        if args.harsanyinet and args.ground_truth:

            '''
                The ground-truth Shapley values could be directly computed by following Definition 1 when n is small.
                To calculate the ground-truth Shapley values, randomly sampling n = 12 variables as input variables 
                in the foreground of the sample x (specifically, on z0).
            '''

            # randomly sample n variables as input variables
            N_PLAYERS = 12
            THRESHOLD = z0.abs().sum(dim=1).max() * 0.1   # roughly define foreground players as players with relatively large values
            foreground = torch.where((z0.abs().sum(dim=1).reshape(-1) > THRESHOLD))[0]
            players = np.random.choice(foreground.cpu().numpy(), N_PLAYERS, replace=False)
            idx = torch.tensor([[i // z0.shape[2], i % z0.shape[3]] for i in players])
            assert len(players) == N_PLAYERS

            # get baseline
            baseline = z0.detach().clone()
            baseline[0, :, idx[:, 0], idx[:, 1]] = 0

            # get the root mean squared error (RMSE)
            Harsanyi_Shap_m = HarsanyiNetShapley_mask(model, z0, label, N_PLAYERS, baseline, players)
            GroundTruth_Shap_m = GroundTruthShapley_mask(model, z0, label, N_PLAYERS, baseline, players, dataset=args.dataset)
            rmse = get_RMSE(GroundTruth_Shap_m, Harsanyi_Shap_m, f"Sample {index}'s HarsanyiNet (1) and Ground truth ({pow(2, N_PLAYERS)})")

            RMSE = RMSE + rmse
            if (index + 1) == args.batch_size:
                RMSE = RMSE / args.batch_size
                print(f"Average RMSE of {args.batch_size} samples: {RMSE}.")







import os
import argparse
import torch
import numpy as np

from shapreg import removal, games, shapley

from model.HarsanyiMLP import HarsanyiNet
from utils.data import get_data_loader
from utils.seed import setup_seed
from utils.tabular.shapley import HarsanyiNetShapley, BruteForceShapley, SamplingShapley, PermutationSamplingShapley, KernelPairShapley, KernelShapley, get_RMSE, get_sample, plot_shapleys

parser = argparse.ArgumentParser(description='Training on Census')
parser.add_argument('--dataset', type=str, default='Census', help=" 'Census', 'Yeast', 'Commercial' can be chosen")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-1)

parser.add_argument('--seed', type=str, default=0)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save_path', type=str, default='./output')

parser.add_argument('--gamma', type=float, default=100, help="a postive value in the tanh function. \
                                                      The larger the gamma, the output of tanh function closer to 1.")
parser.add_argument('--beta', type=int, default=10, help="a postive value for back propagation. In back propagation, \
                                                      the sigmoid function is used to approximate the indicator function.")
parser.add_argument('--num_layers', type=int, default=3, help="number of layers")
parser.add_argument('--n_attributes', type=int, default=12, help="number of input variables")
parser.add_argument('--hidden_dim', type=int, default=100, help="number of channels")
parser.add_argument('--initial_V', type=float, default=1.0, help="initial value for parameter tau")
parser.add_argument('--act_ratio', type= float, default=0.1, help="initial active ratio for children sets.")
parser.add_argument('--comparable_DNN', action='store_true', default=False, help="whether to use a tranditional DNN with comparable size, \
                                                                        False - HarsanyiNet, True - Traditional DNN")


# parameters for attribution
parser.add_argument('--num_samples', type=int, default=10, help="number of samples to be explained")
parser.add_argument('--harsanyinet', action='store_true', default=False, help="whether to use HarsanyiNet to compute Shapley values.")
parser.add_argument('--others', action='store_true', default=False, help="whether to use sampling/kernelshap to compute Shapley values.")
parser.add_argument('--runs', type=int, default=200, help="number of iterations if using Sampling/kernelshap method")  # 2000
parser.add_argument('--test_acc', action='store_true', default=False, help="whether to test the classification accuracy of the model")
parser.add_argument('--plot_shapley', action='store_true', default=False, help="whether to plot shapley errors")
parser.add_argument('--model_path',type=str, default='', help='path of the trained model')
args = parser.parse_args()


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
    args.save_path = os.path.join(args.save_path, str(args.dataset))
    if args.comparable_DNN:
        args.save_path = os.path.join(args.save_path, 'TraditionalDNN')
    print(args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.loss_path = os.path.join(args.save_path, 'AccAndLoss')
    if not os.path.exists(args.loss_path):
        os.makedirs(args.loss_path)
    args.acc_path = os.path.join(args.save_path,'AccAndLoss')
    if not os.path.exists(args.acc_path):
        os.makedirs(args.acc_path)
    if len(args.model_path) == 0:
        args.model_path = os.path.join(args.save_path, 'model_pths',f'epoch{args.epochs-1}.pth')
 
    print("Load model path:", args.model_path)


if __name__ == '__main__':
    init_path(args)
    setup_seed(args.seed)

    train_loader, test_loader, num_classes = get_data_loader(args.dataset, args.batch_size)
    device = args.device if torch.cuda.is_available() else 'cpu'

    model = HarsanyiNet(input_dim=args.n_attributes,
                        num_classes = num_classes,
                        num_layers=args.num_layers,
                        hidden_dim=args.hidden_dim,
                        beta=args.beta, 
                        gamma=args.gamma,
                        initial_V=args.initial_V,
                        act_ratio=args.act_ratio,
                        device=device, 
                        comparable_DNN=args.comparable_DNN,
                        ).to(device)


    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    if args.test_acc:
        test(model, device, test_loader)

    if args.comparable_DNN:
        args.harsanyinet = False
        print("Compute Shapley values on a traditional DNN. Only the sampling method can be used.")
    if args.plot_shapley:
        save_dir =  os.path.join(args.save_path, 'ShapleyResult')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plot_shapleys(args, model, test_loader,device, save_dir)
    else:
        error = 0
        for index in range(args.num_samples):

            # get data
            x_te, y_te = get_sample(test_loader, index=index)
            label = int(y_te)
            x_te = x_te.double()
            model = model.double()
            baseline = torch.zeros_like(x_te).to(device)
        
            print(f"label: {label}, v(empty) = {model(baseline)[:, label].detach().cpu().numpy()}, "
                  f"v(N) - v(empty) = {(model(x_te)[:, label] - model(baseline)[:, label]).detach().cpu().numpy()}")

            # Get the ground truth Shapley value
            Shapley_bf = BruteForceShapley(model, x_te, label)
            # HarsanyiNet to compute Shapley values
            if args.harsanyinet:
                Harsanyi_Shap = HarsanyiNetShapley(model, x_te, label)
                print(f'The Shapley value calculated by HarsanyiMLP is {Harsanyi_Shap}')
                error += get_RMSE(Shapley_bf, Harsanyi_Shap,f"HarsanyiNet (1) and Ground Truth")
            # other post-hoc approximate methods
            if args.others:
                Sampling_Shapley = SamplingShapley(model, x_te, label, args.runs)
                get_RMSE(Shapley_bf, Sampling_Shapley,f"SamplingShapley ({args.runs}) and Ground Truth")
                
                Permutation_Shapley = PermutationSamplingShapley(model,x_te, label, args.runs)
                get_RMSE(Shapley_bf,Permutation_Shapley, f"PermutationSamplingShapley ({args.runs}) and Ground Truth")
           
                Kernel_Shap = KernelShapley(model, x_te, label, args.runs)
                get_RMSE(Shapley_bf, Kernel_Shap, f"KernelSHAP ({args.runs}) and Ground Truth")
           
                Kernel_Shap_Pair = KernelPairShapley(model, x_te, label, args.runs)
                get_RMSE(Shapley_bf, Kernel_Shap_Pair, f"KernelSHAPPair ({args.runs}) and Ground Truth")
        
        if args.harsanyinet:
            print(f'The average RMSE over {args.num_samples} samples of the Shapley value obtained from HarsanyiNet is {error/args.num_samples}')

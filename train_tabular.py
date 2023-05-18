import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from model.HarsanyiMLP import HarsanyiNet
from utils.data import get_data_loader
from utils.plot import plot_loss_acc
from utils.seed import setup_seed

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

args = parser.parse_args()

def train(args,
          model,
          optimizer,
          device,
          train_loader,
          test_loader):

    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []

    for epoch in range(args.epochs):

        t1 = time.time()
        setup_seed(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train
        train_loss_value, train_correct_value = 0, 0
        total_num = 0

        model.train()
        for i, (x_tr, y_tr) in enumerate(train_loader):
            x_tr = x_tr.to(device)
            y_tr = y_tr.to(device)
            optimizer.zero_grad()

            y_pred = model(x_tr)
            loss = criterion(y_pred, y_tr)

            train_loss_value += loss.item() * x_tr.size(0)
            train_correct_value += (y_pred.max(1)[1] == y_tr).sum().item()
            total_num += x_tr.size(0)
            loss.backward()
            optimizer.step()

        avg_tr_loss = train_loss_value / total_num
        avg_tr_acc = train_correct_value / total_num

        train_loss.append(avg_tr_loss)
        train_acc.append(avg_tr_acc)

        print(f"epoch: {epoch} train_loss: {avg_tr_loss:.4f} train_acc: {avg_tr_acc:.4f}")

        # test
        test_loss_value, test_correct_value = 0, 0
        test_total_num = 0
        model.eval()
        for i, (x_te, y_te) in enumerate(test_loader):
            x_te = x_te.to(device)
            y_te = y_te.to(device)
            with torch.no_grad():
                y_pred = model(x_te)
                loss = criterion(y_pred, y_te)

                test_loss_value += loss.item() * x_te.size(0)
                test_correct_value += (y_pred.max(1)[1] == y_te).sum().item()
                test_total_num += x_te.size(0)
        avg_te_loss = test_loss_value / test_total_num
        avg_te_acc = test_correct_value / test_total_num
        test_loss.append(avg_te_loss)
        test_acc.append(avg_te_acc)
        print(f"test_loss: {avg_te_loss:.4f} test_acc: {avg_te_acc:.4f}\n")

        t2 = time.time()
        print(f"time:{t2 - t1}")

        # save model
        if (epoch + 1) % 100 == 0 or (epoch + 1) == args.epochs:
            model_path = os.path.join(args.model_path, f'{args.dataset}.pth')
            torch.save(model.state_dict(), model_path)
            # plot loss and accuracy
            plot_loss_acc(args, train_loss, test_loss, train_acc, test_acc)


def adjust_learning_rate(optimizer, epoch):
    if epoch < 100:
        lr = args.lr
    elif epoch < 200:
        lr = args.lr * 0.1
    else:
        lr = args.lr * (0.1 ** 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_path(args):
    args.save_path = os.path.join(args.save_path, str(args.dataset))
    if args.comparable_DNN:
        args.save_path = os.path.join(args.save_path, str(args.dataset), 'TraditionalDNN')
    print(args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.loss_path = os.path.join(args.save_path, 'AccAndLoss')
    if not os.path.exists(args.loss_path):
        os.makedirs(args.loss_path)
    args.acc_path = os.path.join(args.save_path,'AccAndLoss')
    if not os.path.exists(args.acc_path):
        os.makedirs(args.acc_path)
    args.model_path = os.path.join(args.save_path, 'model_pths')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


if __name__ == '__main__':
    init_path(args)
    setup_seed(args.seed)

    train_loader, test_loader, num_classes = get_data_loader(args.dataset, args.batch_size)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    
    model = HarsanyiNet(input_dim=args.n_attributes,
                        num_classes = num_classes,
                        num_layers=args.num_layers, 
                        beta=args.beta, 
                        gamma=args.gamma,
                        hidden_dim=args.hidden_dim,
                        initial_V=args.initial_V,
                        act_ratio=args.act_ratio,
                        device=device, 
                        comparable_DNN=args.comparable_DNN,
                        ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    t1 = time.time()

    train(args=args,
          model=model,
          optimizer=optimizer,
          device=device,
          train_loader=train_loader,
          test_loader=test_loader)

    t2 = time.time()
    print(f"time:{t2 - t1}")

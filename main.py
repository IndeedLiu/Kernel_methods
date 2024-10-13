import torch
import math
import numpy as np
import os
import pandas as pd
from scipy.stats import chi2
import argparse

from Networks import Vcnet, TR, Drnet
from data import get_iter


def simulate_data(seed=1, nobs=1000, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True):
    np.random.seed(seed)
    X1 = np.random.normal(loc=MX1, scale=1, size=nobs)
    X2 = np.random.normal(loc=MX2, scale=1, size=nobs)
    X3 = np.random.normal(loc=0, scale=1, size=nobs)
    X4 = np.random.normal(loc=MX2, scale=1, size=nobs)
    X5 = np.random.binomial(n=1, p=MX3, size=nobs)

    Z1 = np.exp(X1 / 2)
    Z2 = (X2 / (1 + np.exp(X1))) + 10
    Z3 = (X1 * X3 / 25) + 0.6
    Z4 = (X4 - MX2) ** 2
    Z5 = X5

    muA = 5 * np.abs(X1) + 6 * np.abs(X2) + 3 * np.abs(X5) + np.abs(X4)
    A = (chi2.rvs(df=3, loc=0, scale=1, size=nobs) + muA) / 45

    def true_density_A_X(A, X):
        muA_true = 5 * np.abs(X[:, 0]) + 6 * np.abs(X[:, 1]) + \
            3 * np.abs(X[:, 4]) + np.abs(X[:, 3])
        return chi2.pdf(A, df=3, loc=0, scale=1, nc=muA_true)

    if A_effect:
        Cnum = 1161.25
        Y = -0.15 * A ** 2 + A * (X1 ** 2 + X2 ** 2) - 15 + (X1 + 3) ** 2 + 2 * \
            (X2 - 25) ** 2 + X3 - Cnum + np.random.normal(scale=1, size=nobs)
        Y = Y / 50
        truth = -0.15 * A ** 2 + A * 3.25 - 15
        truth = truth / 50
    else:
        Y = X1 + X1 ** 2 + X2 + X2 ** 2 + X1 * X2 + \
            X5 + np.random.normal(scale=1, size=nobs)
        truth = 5.05

    datz = pd.DataFrame({
        'Y': Y,
        'A': A,
        'Z1': Z1,
        'Z2': Z2,
        'Z3': Z3,
        'Z4': Z4,
        'Z5': Z5,
        'truth': truth
    })

    datx = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5
    })

    return {
        'data': datz,
        'true_adrf': truth,
        'original_covariates': datx,
        'true_density_A_X': true_density_A_X
    }


def curve(model, test_matrix, t_grid, targetreg=None):
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]
    test_loader = get_iter(
        test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg is None:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            out = out[1].data.squeeze()
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() -
               t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse
    else:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(t, x)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() -
               t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse


def adjust_learning_rate(optimizer, init_lr, epoch, lr_type, num_epoch):
    if lr_type == 'cos':
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, state['model'] + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean()


def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with simulated data')

    # i/o
    parser.add_argument('--data_dir', type=str,
                        default='dataset/simu1/eval/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str,
                        default='logs/simu1/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100,
                        help='print train info freq')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True,
                        help='whether to plot adrf curves. (only run two methods if set true; the label of fig is only for drnet and vcnet in a certain order)')

    args = parser.parse_args()

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9
    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate simulated data
    simulated_data = simulate_data(
        seed=1, nobs=1000, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True)
    data = simulated_data['data'].to_numpy()
    # Getting original X1 to X5 data
    datx = simulated_data['original_covariates'].to_numpy()

    n_train = int(0.75 * data.shape[0])
    train_data = data[:n_train]
    test_data = data[n_train:]
    train_x = datx[:n_train]  # Train part of the original covariates
    test_x = datx[n_train:]  # Test part of the original covariates

    # Process train_matrix and test_matrix
    train_matrix = torch.from_numpy(
        np.hstack([train_data[:, [1]], train_x, train_data[:, [0]]])).float()  # A, X1, X2, X3, X4, X5, Y
    test_matrix = torch.from_numpy(
        np.hstack([test_data[:, [1]], test_x, test_data[:, [0]]])).float()  # A, X1, X2, X3, X4, X5, Y

    # Process t_grid
    # Assuming 'truth' is the last column of the datz DataFrame
    truth_values = test_data[:, -1]
    t_grid = torch.stack([torch.from_numpy(test_data[:, 1]), torch.from_numpy(
        truth_values)], dim=0).float()  # A, truth

    print('data', train_matrix.shape, test_matrix.shape, t_grid.shape)
    print(train_matrix.shape[0], train_matrix.shape[1])

    train_loader = get_iter(train_matrix, batch_size=500, shuffle=True)
    test_loader = get_iter(
        test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    grid = []
    MSE = []

    # rest of your training and evaluation code

    # choose from {'Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr'}
    method_list = ['Drnet', 'Vcnet']

    for model_name in method_list:
        # import model
        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(5, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif model_name == 'Drnet' or model_name == 'Drnet_tr':
            cfg_density = [(5, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
            cfg_density = [(5, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        # use Target Regularization?
        if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Tarnet_tr':
            isTargetReg = 1
        else:
            isTargetReg = 0

        if isTargetReg:
            tr_knots = list(np.arange(0.1, 1, 0.1))
            tr_degree = 2
            TargetReg = TR(tr_degree, tr_knots)
            TargetReg._initialize_weights()

        # best cfg for each model
        if model_name == 'Tarnet':
            init_lr = 0.05
            alpha = 1.0
        elif model_name == 'Tarnet_tr':
            init_lr = 0.05
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Drnet':
            init_lr = 0.01
            alpha = 1.
        elif model_name == 'Drnet_tr':
            init_lr = 0.02
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Vcnet':
            init_lr = 0.00005
            alpha = 0.5
        elif model_name == 'Vcnet_tr':
            init_lr = 0.0001
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.

        optimizer = torch.optim.SGD(model.parameters(
        ), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

        if isTargetReg:
            tr_optimizer = torch.optim.SGD(
                TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

        print('model = ', model_name)
        for epoch in range(num_epoch):

            for idx, (inputs, y) in enumerate(train_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]

                if isTargetReg:
                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    trg = TargetReg(t)
                    loss = criterion(out, y, alpha=alpha) + \
                        criterion_TR(out, trg, y, beta=beta)
                    loss.backward()
                    optimizer.step()

                    tr_optimizer.zero_grad()
                    out = model.forward(t, x)
                    trg = TargetReg(t)
                    tr_loss = criterion_TR(out, trg, y, beta=beta)
                    tr_loss.backward()
                    tr_optimizer.step()
                else:
                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    loss = criterion(out, y, alpha=alpha)
                    loss.backward()
                    optimizer.step()

            if epoch % verbose == 0:
                print('current epoch: ', epoch)
                print('loss: ', loss.data)

        if isTargetReg:
            t_grid_hat, mse = curve(
                model, test_matrix, t_grid, targetreg=TargetReg)
        else:
            t_grid_hat, mse = curve(model, test_matrix, t_grid)

        mse = float(mse)
        print('current loss: ', float(loss.data))
        print('current test loss: ', mse)
        print('-----------------------------------------------------------------')

        save_checkpoint({
            'model': model_name,
            'best_test_loss': mse,
            'model_state_dict': model.state_dict(),
            'TR_state_dict': TargetReg.state_dict() if isTargetReg else None,
        }, checkpoint_dir=save_path)

        print('-----------------------------------------------------------------')

        grid.append(t_grid_hat)
        MSE.append(mse)

    if args.plt_adrf:
        import matplotlib.pyplot as plt

        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 22,
                 }

        font_legend = {'family': 'Times New Roman',
                       'weight': 'normal',
                       'size': 22,
                       }
        plt.figure(figsize=(5, 5))

        c1 = 'gold'
        c2 = 'red'
        c3 = 'dodgerblue'

        truth_grid = t_grid[:, t_grid[0, :].argsort()]
        x = truth_grid[0, :]
        y = truth_grid[1, :]
        plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)

        x = grid[1][0, :]
        y = grid[1][1, :]
        plt.scatter(x, y, marker='h', label='Vcnet',
                    alpha=1, zorder=2, color=c2, s=20)

        x = grid[0][0, :]
        y = grid[0][1, :]
        plt.scatter(x, y, marker='H', label='Drnet',
                    alpha=1, zorder=3, color=c3, s=20)

        plt.yticks(np.arange(-2.0, 1.1, 0.5),
                   fontsize=0, family='Times New Roman')
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=0,
                   family='Times New Roman')
        plt.grid()
        plt.legend(prop=font_legend, loc='lower left')
        plt.xlabel('Treatment', font1)
        plt.ylabel('Response', font1)

        plt.show()

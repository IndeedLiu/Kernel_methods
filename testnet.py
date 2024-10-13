import torch
import math
import numpy as np
import os
import pandas as pd

from Networks import Vcnet, Drnet, TR
from data import get_iter


import argparse
import numpy as np
import torch
import pandas as pd
from scipy.stats import chi2


def simulate_data(seed=1, nobs=1000, MX1=-0.5, MX2=1, MX3=0.3, A_effect=True):
    np.random.seed(seed)

    # Generate covariates and dose
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

    A = chi2.rvs(df=3, loc=0, scale=1, size=nobs) + muA

    def true_density_A_X(A, X):
        muA_true = 5 * np.abs(X[:, 0]) + 6 * np.abs(X[:, 1]) + \
            3 * np.abs(X[:, 4]) + np.abs(X[:, 3])
        return chi2.pdf(A, df=3, loc=0, scale=1, nc=muA_true)

    if A_effect:
        Cnum = 1161.25
        Y = -0.15 * A ** 2 + A * (X1 ** 2 + X2 ** 2) - 15 + (X1 + 3) ** 2 + 2 * (
            X2 - 25) ** 2 + X3 - Cnum + np.random.normal(scale=1, size=nobs)
        Y = Y / 50

        truth = -0.15 * A ** 2 + A * 0.065 - 15
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


def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':  # cos without warm-up
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
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion


def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()


def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str,
                        default='dataset/simu1/eval/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str,
                        default='logs/simu1/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=800,
                        help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100,
                        help='print train info freq')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                        'the label of fig is only for drnet and vcnet in a certain order)')

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

    # Simulate data
    simulated_data = simulate_data()
    datz = simulated_data['data']
    datx = simulated_data['original_covariates']
    truth = simulated_data['true_adrf']

    # Convert simulated data to torch tensors
    train_data = datz.sample(frac=0.8, random_state=1)
    test_data = datz.drop(train_data.index)

    train_matrix = torch.tensor(
        train_data[['A', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Y']].values).float()
    test_matrix = torch.tensor(
        test_data[['A', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Y']].values).float()
    t_grid = torch.tensor(np.vstack(
        (train_data['A'].values, truth[:train_data.shape[0]])).astype(np.float32))

    train_loader = get_iter(train_matrix, batch_size=500, shuffle=True)
    test_loader = get_iter(
        test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    grid = []
    MSE = []

    # choose from {'Tarnet', 'Tarnet_tr', 'Vcnet', 'Vcnet_tr', 'Drnet', 'Drnet_tr'}
    model_name = 'Vcnet'

    # run for Drnet
    if model_name == 'Drnet':
        model = Drnet()
        init_lr = 1e-3
        optimizer = torch.optim.Adam(
            model.parameters(), lr=init_lr, weight_decay=wd)
        for epoch in range(num_epoch):
            lr = adjust_learning_rate(optimizer, init_lr, epoch)

            model.train()
            for idx, (inputs, y) in enumerate(train_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                out = model.forward(t, x)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % verbose == 0:
                with torch.no_grad():
                    model.eval()
                    for idx, (inputs, y) in enumerate(test_loader):
                        t = inputs[:, 0]
                        x = inputs[:, 1:]
                        out = model.forward(t, x)
                        loss = criterion(out, y)
                    print('Epoch: {}, Test Loss: {:.4f}, Learning Rate: {:.4f}'.format(
                        epoch, loss.item(), lr))

                    t_grid_hat, mse = curve(model, test_matrix, t_grid)
                    grid.append(t_grid_hat)
                    MSE.append(mse)
                    print('Test MSE: {:.4f}'.format(mse))

    # run for Drnet_tr
    if model_name == 'Drnet_tr':
        model = Drnet()
        targetreg = TR()
        init_lr = 1e-3
        tr_init_lr = 1e-3

        optimizer = torch.optim.Adam(
            model.parameters(), lr=init_lr, weight_decay=wd)
        tr_optimizer = torch.optim.Adam(
            targetreg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

        for epoch in range(num_epoch):
            lr = adjust_learning_rate(optimizer, init_lr, epoch)
            tr_lr = adjust_learning_rate(tr_optimizer, tr_init_lr, epoch)

            model.train()
            targetreg.train()

            for idx, (inputs, y) in enumerate(train_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                trg = targetreg(t)

                out = model.forward(t, x)
                loss = criterion_TR(out, trg, y)
                tr_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_optimizer.step()
            if epoch % verbose == 0:
                with torch.no_grad():
                    model.eval()
                    targetreg.eval()
                    for idx, (inputs, y) in enumerate(test_loader):
                        t = inputs[:, 0]
                        x = inputs[:, 1:]
                        trg = targetreg(t)
                        out = model.forward(t, x)
                        loss = criterion_TR(out, trg, y)
                    print('Epoch: {}, Test Loss: {:.4f}, Learning Rate: {:.4f}, TR Learning Rate: {:.4f}'.format(
                        epoch, loss.item(), lr, tr_lr))

                    t_grid_hat, mse = curve(
                        model, test_matrix, t_grid, targetreg=targetreg)
                    grid.append(t_grid_hat)
                    MSE.append(mse)
                    print('Test MSE: {:.4f}'.format(mse))

    # run for Vcnet
    if model_name == 'Vcnet':
        model = Vcnet()
        init_lr = 1e-3
        optimizer = torch.optim.Adam(
            model.parameters(), lr=init_lr, weight_decay=wd)
        for epoch in range(num_epoch):
            lr = adjust_learning_rate(optimizer, init_lr, epoch)

            model.train()
            for idx, (inputs, y) in enumerate(train_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                out = model.forward(t, x)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % verbose == 0:
                with torch.no_grad():
                    model.eval()
                    for idx, (inputs, y) in enumerate(test_loader):
                        t = inputs[:, 0]
                        x = inputs[:, 1:]
                        out = model.forward(t, x)
                        loss = criterion(out, y)
                    print('Epoch: {}, Test Loss: {:.4f}, Learning Rate: {:.4f}'.format(
                        epoch, loss.item(), lr))

                    t_grid_hat, mse = curve(model, test_matrix, t_grid)
                    grid.append(t_grid_hat)
                    MSE.append(mse)
                    print('Test MSE: {:.4f}'.format(mse))

    # run for Vcnet_tr
    if model_name == 'Vcnet_tr':
        model = Vcnet()
        targetreg = TR()
        init_lr = 1e-3
        tr_init_lr = 1e-3

        optimizer = torch.optim.Adam(
            model.parameters(), lr=init_lr, weight_decay=wd)
        tr_optimizer = torch.optim.Adam(
            targetreg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

        for epoch in range(num_epoch):
            lr = adjust_learning_rate(optimizer, init_lr, epoch)
            tr_lr = adjust_learning_rate(tr_optimizer, tr_init_lr, epoch)

            model.train()
            targetreg.train()

            for idx, (inputs, y) in enumerate(train_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                trg = targetreg(t)

                out = model.forward(t, x)
                loss = criterion_TR(out, trg, y)
                tr_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_optimizer.step()
            if epoch % verbose == 0:
                with torch.no_grad():
                    model.eval()
                    targetreg.eval()
                    for idx, (inputs, y) in enumerate(test_loader):
                        t = inputs[:, 0]
                        x = inputs[:, 1:]
                        trg = targetreg(t)
                        out = model.forward(t, x)
                        loss = criterion_TR(out, trg, y)
                    print('Epoch: {}, Test Loss: {:.4f}, Learning Rate: {:.4f}, TR Learning Rate: {:.4f}'.format(
                        epoch, loss.item(), lr, tr_lr))

                    t_grid_hat, mse = curve(
                        model, test_matrix, t_grid, targetreg=targetreg)
                    grid.append(t_grid_hat)
                    MSE.append(mse)
                    print('Test MSE: {:.4f}'.format(mse))

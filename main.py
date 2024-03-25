import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from FedDR.synFedDR import syn_feddr
from FedQuasiNewton.FedQuasiNewton import FedQuasiNewton
from utils.generate_dataset import get_data
from utils.swith_variable_type import dict_swtich_order


# 初始化
def init_para(item=('synthetic', 'mnist')):
    init_paras = {}
    if item == 'synthetic':
        dimension = 60
        n_class = 10
    elif item == 'mnist':
        dimension = 784
        n_class = 10
    elif item == 'femnist':
        dimension = 784
        n_class = 62
    else:
        dimension = None
        n_class = None
    torch.manual_seed(123)
    init_paras['predict.weight'] = torch.normal(0, 1, (n_class, dimension))
    init_paras['predict.bias'] = torch.zeros(n_class)
    init_paras = dict_swtich_order(init_paras)

    return init_paras


def plot_figure(item, dataset, lambd):
    figure_class = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
    plt.figure(figsize=(20, 10))
    plt.suptitle(dataset, fontsize=40)
    for i in range(len(figure_class)):
        plt.subplot(2, 2, i + 1)
        plt.plot(item[i][0], color='red')
        plt.plot(item[i][1], color='blue')
        plt.xlabel('Round')

        plt.title(label=figure_class[i], fontsize=10)
        plt.legend(labels=['FedQuasiNewton', 'FedDR'], loc="best", fontsize=10)
    plt.savefig('./img/{}-lambda={}.png'.format(dataset, lambd))
    plt.show()


if __name__ == '__main__':
    client_num = 30
    item = 'synthetic'           # synthetic, mnist, femnist
    dataset = 'synthetic_iid'        # synthetic_iid, synthetic_0_0, synthetic_0.5_0.5, synthetic_1_1, mnist
    max_iter = 200
    reg_type = 'l1'
    lambd = 0.001
    active_ratio = 1 / 4

    train_data, global_train_data, global_test_data = get_data(client_num, dataset_item=dataset)
    init_parameters = init_para(item=item)

    #################### FedQuasiNewton ########################
    options = {
        'loss_func': nn.CrossEntropyLoss(),
        'rounds': max_iter,
        'local_epoch': 1,
        'regularizer': reg_type,
        'lambd': lambd,
        'data_item': item,
        # client, for ADMM
        'alpha': 1,
        'rho': 5,  # 5
        # hessian
        'global_hessian': None,
        'nu_hat': 10, # 10
        'eta': 0.99,
        'active_ratio': active_ratio
    }
    print('---------------------------FedQuasiNewton-------------------------------')
    para1, train_loss_admm1, train_acc_admm1, test_loss_admm1, test_acc_admm1 = \
        FedQuasiNewton(train_data, global_test_data, client_num, init_parameters, **options)

    ########################## FedDR ############################
    options_dr = {
        'loss_func': nn.CrossEntropyLoss(),
        'rounds': max_iter,
        'regularizer': reg_type,
        'lambd': lambd,
        'data_item': item,
        'local_epoch': 20,
        'batch_size': 50,
        'learning_rate': 0.01,
        'eta': 500,
        'alpha': 1.95,
        'active_client_ratio': active_ratio
    }
    print('------------------------FedDR---------------------------')
    para_dr, train_loss_dr, train_acc_dr, test_loss_dr, test_acc_dr = \
        syn_feddr(train_data, global_train_data, global_test_data, client_num, init_parameters, **options_dr)

    # plot
    train_loss = [train_loss_admm1, train_loss_dr]
    train_acc = [train_acc_admm1, train_acc_dr]
    test_loss = [test_loss_admm1, test_loss_dr]
    test_acc = [test_acc_admm1, test_acc_dr]

    plot_figure([train_loss, train_acc, test_loss, test_acc], dataset, options['lambd'])

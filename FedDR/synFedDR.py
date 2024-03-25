import copy
import torch
import numpy as np
from utils.swith_variable_type import dict_swtich_order
from utils.model import Net
from FedDR.input import set_options
from FedDR.clients_dr import Client_DR, evaluate
from FedDR.server import Server_DR


def syn_feddr(train_data, global_train_data, global_test_data, client_num, init_para, **options):
    options = set_options(**options)
    if options.data_item == 'synthetic':
        dimension = 60
        n_class = 10
    elif options.data_item == 'mnist':
        dimension = 784
        n_class = 10
    elif options.data_item == 'femnist':
        dimension = 784
        n_class = 62
    global_model = Net(dimension, n_class)
    model = Net(dimension, n_class)
    # Initiation clients
    init_zeros = {k: torch.zeros_like(init_para[k]) for k in init_para}
    init_zeros = dict_swtich_order(init_zeros)

    local_yi = [init_para for _ in range(client_num)]
    local_xi = [init_zeros for _ in range(client_num)]
    local_xhat = [init_zeros for _ in range(client_num)]
    # Initiation server
    global_parameters = copy.deepcopy(init_para)
    global_model.load_state_dict(global_parameters)
    for i in range(client_num):
        local_model = Client_DR(train_data, i, options)
        local_xi[i] = local_model.update_xi(global_parameters, local_yi[i], model=copy.deepcopy(model))
        local_xhat[i] = local_model.update_xhat(local_xi[i], local_yi[i])

    # store train loss, acc., test loss, acc.
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # loss and accuracy
    round_train_loss, round_train_acc = evaluate(global_train_data, global_model, options)
    round_test_loss, round_test_acc = evaluate(global_test_data, global_model, options)
    train_loss.append(round_train_loss)
    train_acc.append(round_train_acc)
    test_loss.append(round_test_loss)
    test_acc.append(round_test_acc)

    print("round: %d, train loss: %4f, train acc.: %4f, test loss: %4f, test acc.:%4f"
          % (0, round_train_loss, round_train_acc, round_test_loss, round_test_acc))

    # number of active clients
    m = max(int(client_num * options.C), 1)
    # begin update
    for iteration in range(options.rounds):
        np.random.seed(123 + iteration)
        active_clients = np.random.choice(range(client_num), m, replace=False)
        global_para = copy.deepcopy(global_parameters)
        for idx in active_clients:
            local_model = Client_DR(train_data, idx, options)
            local_yi[idx] = local_model.update_yi(global_para, local_xi[idx], local_yi[idx])
            local_xi[idx] = local_model.update_xi(global_para, local_yi[idx], model=copy.deepcopy(model))
            local_xhat[idx] = local_model.update_xhat(local_xi[idx], local_yi[idx])

        # server
        server_process = Server_DR(client_num, options)
        global_xtilde = server_process.aggregate(local_xhat)
        if options.lambd == 0:
            global_parameters = global_xtilde
        else:
            global_parameters = server_process.proximal(global_xtilde)
        global_model.load_state_dict(global_parameters)

        # loss and accuracy
        round_train_loss, round_train_acc = evaluate(global_train_data, global_model, options)
        round_test_loss, round_test_acc = evaluate(global_test_data, global_model, options)
        train_loss.append(round_train_loss)
        train_acc.append(round_train_acc)
        test_loss.append(round_test_loss)
        test_acc.append(round_test_acc)

        print("round: %d, train loss: %4f, train acc.: %4f, test loss: %4f, test acc.:%4f"
              % (iteration+1, round_train_loss, round_train_acc, round_test_loss, round_test_acc))
    return global_parameters, train_loss, train_acc, test_loss, test_acc

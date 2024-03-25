import torch
import numpy as np
from .input import set_options
from utils.model import Net
from .clients import Clients, test_result
from .server import Server


def FedQuasiNewton(train_data, test_data, client_num, init_para, **options):
    options = set_options(**options)
    # set Network
    if options.data_item == 'synthetic':
        dimension = 60
        n_class = 10
    elif options.data_item == 'mnist':
        dimension = 784
        n_class = 10
    elif options.data_item == 'femnist':
        dimension = 784
        n_class = 62
    dim = dimension * n_class + n_class
    global_model = Net(dimension, n_class)
    # Initiation
    global_parameters = init_para
    global_direction = torch.zeros(dim)
    local_delta_d_list = [torch.zeros(dim) for _ in range(client_num)]
    local_lambda_list = [torch.zeros(dim) for _ in range(client_num)]
    global_model.load_state_dict(global_parameters)
    # store what we need
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    #  calculate the initial train_loss and train_acc
    round_train_loss, round_train_acc, round_sample_num = 0.0, 0.0, 0.0
    for c in range(client_num):
        local_model = Clients(train_data, c, global_model, 0, local_batch=20, dim=dim, options=options)
        client_acc, client_loss, client_sample_num = local_model.caculate_train_acc_loss(global_model)
        round_train_loss += client_loss * client_sample_num
        round_train_acc += client_acc * client_sample_num
        round_sample_num += client_sample_num
    round_train_loss /= round_sample_num
    round_train_acc /= round_sample_num

    train_loss.append(round_train_loss)
    train_acc.append(round_train_acc)
    # test acc and test loss
    round_test_acc, round_test_loss = test_result(global_model, options.loss_func, test_data)
    test_acc.append(round_test_acc)
    test_loss.append(round_test_loss)

    print(
        "round: 0, train loss: %4f, train acc: %4f, test loss: %4f, test acc: %4f"
        % (round_train_loss, round_train_acc, round_test_loss, round_test_acc))

    global_para_old = global_parameters
    # number of active clients
    m = max(int(options.active_ratio * client_num), 1)
    for iteration in range(options.rounds):
        # sample active clients
        np.random.seed(123+iteration)
        active_user = np.random.choice(client_num, m, replace=False)
        # run ADMM several times
        for epoch in range(options.local_epoch):
            U = []
            # store the local grad dif for server
            local_grad_dif_list = []
            # client update
            for idx in active_user:
                local_model = Clients(train_data, idx, global_model, iteration,
                                      local_batch=train_data["num_samples"][idx], dim=dim, options=options)
                delta_d, local_lambda, dif_grad, B_0, u = local_model.client_update(global_parameters,
                                                global_para_old, global_direction, local_lambda_list[idx])
                # store
                local_delta_d_list[idx] = delta_d
                local_grad_dif_list.append(dif_grad)
                local_lambda_list[idx] = local_lambda
                U.append(u / np.sqrt(m))

            # server update
            global_direction = Server.aggregate_direction(local_delta_d_list)

        U = torch.t(torch.vstack(U))

        server_process = Server(iteration, global_parameters, global_para_old, B_0, U, options)
        # store the global parameters for next iteration to generate the hessian
        global_para_old = global_parameters
        if options.lambd == 0:  # without regularize
            global_parameters = server_process.update_inprox(global_direction)
        else:
            global_parameters = server_process.prox_calculate(local_grad_dif_list, global_direction)
        # update the model
        global_model.load_state_dict(global_parameters)

        #  calculate train_loss and train_acc
        round_train_loss, round_train_acc, round_sample_num = 0.0, 0.0, 0.0
        for c in range(client_num):
            local_model = Clients(train_data, c, global_model, iteration, local_batch=20, dim=dim, options=options)
            client_acc, client_loss, client_sample_num = local_model.caculate_train_acc_loss(global_model)
            round_train_loss += client_loss * client_sample_num
            round_train_acc += client_acc * client_sample_num
            round_sample_num += client_sample_num
        round_train_loss /= round_sample_num
        round_train_acc /= round_sample_num

        train_loss.append(round_train_loss)
        train_acc.append(round_train_acc)
        # test acc and test loss
        round_test_acc, round_test_loss = test_result(global_model, options.loss_func, test_data)
        test_acc.append(round_test_acc)
        test_loss.append(round_test_loss)

        print(
            "round: %d, train loss: %4f, train acc: %4f, test loss: %4f, test acc: %4f"
            % (iteration+1, round_train_loss, round_train_acc, round_test_loss, round_test_acc))
    return global_parameters, train_loss, train_acc, test_loss, test_acc

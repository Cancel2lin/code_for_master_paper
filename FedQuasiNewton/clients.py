import copy

import torch
from torch.utils.data import DataLoader

from utils.deal_dataset import DataSplit, Global_Data
from utils.generate_Hessian import Hessian_SR1
from utils.swith_variable_type import mat_switch_vec


class Clients(object):
    def __init__(self, dataset, idx, model, iteration, local_batch, dim, options):
        """
        dim = dimension * n_class + n_class
        options(required):
            .'alpha'    for local ADMM
            .'rho'      for local ADMM, in argument Lagrange function
            .'hessian'  ={True,False}, do we need to build the oral hessian matrix
        """
        self.options = options
        self.train_data = DataLoader(DataSplit(dataset, idx), batch_size=local_batch, shuffle=True)
        self.iteration = iteration
        self.rho = options.rho
        self.alpha = options.alpha
        self.round = options.rounds
        self.loss_func = options.loss_func
        self.model = model
        self.dim = dim

    @staticmethod
    def quasi_hessian(iteration, global_parameters, global_para_old, local_grad, local_grad_old, options):
        para_new = []
        para_old = []
        for i in global_parameters.keys():
            para_new.append(mat_switch_vec(global_parameters[i]))
            para_old.append(mat_switch_vec(global_para_old[i]))
        para_new = torch.hstack(para_new)
        para_old = torch.hstack(para_old)

        dif_grad = local_grad - local_grad_old

        B_k, B_0, u = Hessian_SR1(iteration, para_new, para_old, dif_grad, options.nu_hat, options.eta)

        return B_k, B_0, u

    def compute_hessian_grad(self, global_parameters, global_para_old):
        model = copy.deepcopy(self.model)
        model_old = copy.deepcopy(self.model)
        model.load_state_dict(global_parameters)
        model_old.load_state_dict(global_para_old)
        # calculate the gradient and hessian
        model.zero_grad()
        model_old.zero_grad()
        for _, (batch_x, batch_y) in enumerate(self.train_data):
            batch_output = model(batch_x.float())
            batch_output_old = model_old(batch_x.float())
            batch_loss = self.loss_func(batch_output, batch_y.type(torch.LongTensor))
            batch_loss_old = self.loss_func(batch_output_old, batch_y.type(torch.LongTensor))
        # calculate gradient
        batch_loss.backward()
        batch_loss_old.backward()
        grad = []
        grad_old = []
        for _, parms in model.named_parameters():
            grad.append(mat_switch_vec(parms.grad))
        grad = torch.hstack(grad)
        for _, parms in model_old.named_parameters():
            grad_old.append(mat_switch_vec(parms.grad))
        grad_old = torch.hstack(grad_old)
        hessian_mat, B_0, u = Clients.quasi_hessian(self.iteration, global_parameters, global_para_old, grad,
                                                    grad_old, self.options)

        dif_grad = grad - grad_old
        return grad, hessian_mat, B_0, u, dif_grad

    def client_update(self, global_parameters, global_para_old, global_direction, local_lambda_old):
        """
        global_parameters: Ordereddict() type
        global_direction: The global ADMM direction; long tensor type: tensor('w','b') 'w'.shape=[1, dimension*n_class]
        local_direction: The local primal parameters, the local ADMM direction; long tensor type
        local_grad_old: The last local gradient; long tensor type
        local_lambda_old: The last local dual parameters; long tensor type
        """
        # calculate gradient and Hessian matrix
        grad, hessian_mat, B_0, u, dif_grad = Clients.compute_hessian_grad(self, global_parameters, global_para_old)
        # update the local primal parameters---the local ADMM direction
        A = hessian_mat + (self.alpha + self.rho) * torch.eye(self.dim)
        b = grad - local_lambda_old + self.rho * global_direction
        local_direction = torch.linalg.solve(A, b)
        # update the local dual parameters
        local_lambda = local_lambda_old + self.rho * (local_direction - global_direction)
        # for global direction updated
        delta_d = local_direction + local_lambda / self.rho

        # delta_d for global direction update, dif_grad for generating server's hessian or
        # B_0, u for server's proximal compute.
        # local_lambda for next update.
        return delta_d, local_lambda, dif_grad, B_0, u

    def caculate_train_acc_loss(self, model):
        loss, sample_num, correct = 0.0, 0.0, 0.0

        for batch_idx, (batch_x, batch_y) in enumerate(self.train_data):
            batch_output = model(batch_x.float())
            batch_loss = self.loss_func(batch_output, batch_y.type(torch.LongTensor))
            loss += batch_loss.item() * len(batch_y)

            y_pred = torch.max(batch_output, 1)[1]
            y_pred = y_pred.view(-1)

            correct += torch.sum(torch.eq(y_pred, batch_y)).item()
            sample_num += len(batch_y)

        acc = correct / sample_num
        train_loss = loss / sample_num

        return acc, train_loss, sample_num


def test_result(model, loss_func, test_data):
    loss, sample_num, correct = 0.0, 0.0, 0.0
    test_dataset = DataLoader(Global_Data(test_data), batch_size=20, shuffle=False)

    for batch_idx, (batch_x, batch_y) in enumerate(test_dataset):
        batch_output = model(batch_x)
        batch_loss = loss_func(batch_output, batch_y.type(torch.LongTensor))
        loss += batch_loss.item() * len(batch_y)

        y_pred = torch.max(batch_output, 1)[1]
        y_pred = y_pred.view(-1)

        correct += torch.sum(torch.eq(y_pred, batch_y)).item()
        sample_num += len(batch_y)

    test_acc = correct / sample_num
    test_loss = loss / sample_num

    return test_acc, test_loss

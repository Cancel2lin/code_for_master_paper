import torch
import torch.nn

from regularizer.proximal_operator import proximal_compute
from utils.generate_Hessian import Hessian_SR1
from utils.swith_variable_type import mat_switch_vec, vec_swith_mat, dict_swtich_order
from .input import nonsmooth


class Server(object):
    def __init__(self, iteration, global_parameters, global_para_old, B_0, U, options):
        """
        global_para_old: Ordereddict() for construct global hessian matrix
        mu: for proximal
        options(required):
            .'regularizer'  the regularize, such as "l1"
            .'lambd'        the regularize coefficient
            .'rho','alpha'  for ADMM
            .'nu_hat','eta' for generate hessian

        """
        self.options = options
        self.iteration = iteration
        self.global_parameters = global_parameters
        self.global_para_old = global_para_old
        self.regularizer = nonsmooth(options.h)
        self.lambd = options.lambd
        self.rho = options.rho
        self.alpha = options.alpha
        self.nu_hat = options.nu_hat
        self.eta = options.eta
        self.global_hessian = options.global_hessian
        self.B_0 = B_0
        self.U = U

    @staticmethod
    def aggregate_direction(local_direction):
        """
        local_direction=[torch.tensor('w','b') for _ in range(client_num)]
        """
        global_direction = torch.zeros_like(local_direction[0], dtype=torch.float32)
        for i in range(len(local_direction)):
            global_direction += local_direction[i]
        global_direction /= len(local_direction)

        return global_direction

    def update_inprox(self, global_direction):
        global_para = []
        for i in self.global_parameters.keys():
            global_para.append(mat_switch_vec(self.global_parameters[i]))
            if i == list(self.global_parameters.keys())[0]:
                dim_w = len(mat_switch_vec(self.global_parameters[i]))
        global_para = torch.hstack(global_para)
        inner_parameters = global_para - global_direction
        if self.lambd == 0:
            para_weight = inner_parameters[:dim_w]
            para_bias = inner_parameters[dim_w:]
            para_weight = vec_swith_mat(para_weight, len(para_bias))
            inner_parameters = {'predict.weight': para_weight, 'predict.bias': para_bias}
            inner_parameters = dict_swtich_order(inner_parameters)

        return inner_parameters

    def prox_calculate(self, local_grad_dif, global_direction):
        """
        local_grad_dif = [torch.tensor('w','b') for _ in range(client_num)]
        """
        para_new = []
        para_old = []
        for i in self.global_parameters.keys():
            para_new.append(mat_switch_vec(self.global_parameters[i]))
            para_old.append(mat_switch_vec(self.global_para_old[i]))
            if i == list(self.global_parameters.keys())[0]:
                dim_w = len(mat_switch_vec(self.global_parameters[i]))
        para_new = torch.hstack(para_new)
        para_old = torch.hstack(para_old)

        global_grad_dif = torch.zeros_like(local_grad_dif[0], dtype=torch.float32)
        for i in range(len(local_grad_dif)):
            global_grad_dif += local_grad_dif[i]
        global_grad_dif /= len(local_grad_dif)
        active_num = len(local_grad_dif)
        # generate hessian matrix
        if self.global_hessian == 'SR1': # could not work well, don't use this.
            B_k, B_0, u = Hessian_SR1(self.iteration, para_new, para_old, global_grad_dif, nu_hat=self.nu_hat,
                                      eta=self.eta)
            U = [u]
            U = torch.t(torch.vstack(U))
        else:  # the method we used in our paper.
            B_0 = self.B_0
            U = self.U
            AssertionError(torch.linalg.matrix_rank(U) == active_num)   
        # calculate the proximal
        inner_para = self.update_inprox(global_direction)
        prox = self.regularizer.prox
        partialprox = self.regularizer.partialprox
        proximal_result = proximal_compute(inner_para, prox, partialprox, B_0, U, self.lambd)

        global_parameters = proximal_result
        global_para_weight = global_parameters[:dim_w]
        global_para_bias = global_parameters[dim_w:]
        global_para_weight = vec_swith_mat(global_para_weight, len(global_para_bias))
        global_parameters = {'predict.weight': global_para_weight, 'predict.bias': global_para_bias}
        global_parameters = dict_swtich_order(global_parameters)

        return global_parameters

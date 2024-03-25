import copy
import torch

from FedDR.input import nonsmooth


class Server_DR(object):
    def __init__(self, client_num, options):
        self.regularizer = nonsmooth(options.h)
        self.lambd = options.lambd
        self.eta = options.eta
        self.client_num = client_num

    @staticmethod
    def aggregate(local_xhat):
        xtilde = copy.deepcopy(local_xhat[0])
        for i in local_xhat[0].keys():
            tmp = torch.zeros_like(local_xhat[0][i])
            for k in range(len(local_xhat)):
                tmp += local_xhat[k][i]
            tmp = torch.true_divide(tmp, len(local_xhat))
            xtilde[i].copy_(tmp)
        return xtilde

    def proximal(self, global_xtilde):
        prox = self.regularizer.prox
        global_parameters = copy.deepcopy(global_xtilde)
        for i in global_xtilde.keys():
            global_parameters[i] = prox(global_xtilde[i], self.eta * self.lambd)

        return global_parameters


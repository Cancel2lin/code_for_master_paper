from regularizer import regularizer
import torch.nn as nn


class set_options:
    def __init__(self, **options):
        # using gpu or not
        options.setdefault("use_gpu", False)
        self.use_gpu = options['use_gpu']
        # loss function
        options.setdefault("loss_func", nn.CrossEntropyLoss())
        self.loss_func = options['loss_func']
        # iteration times
        options.setdefault("rounds", 200)
        self.rounds = options['rounds']
        # regularizer
        options.setdefault("regularizer", "l1")
        self.h = options['regularizer']
        options.setdefault("lambd", 0.01)
        self.lambd = options['lambd']
        # data
        options.setdefault("data_item", "synthetic")
        self.data_item = options['data_item']
        # for Hessian generate
        options.setdefault("nu_hat", 1e-6)
        self.nu_hat = options['nu_hat']
        options.setdefault("eta", 0.9)
        self.eta = options['eta']
        options.setdefault("global_hessian", None)
        self.global_hessian = options['global_hessian']
        # for ADMM
        options.setdefault("local_epoch", 1)
        self.local_epoch = options['local_epoch']
        options.setdefault("alpha", 0.0)
        self.alpha = options['alpha']
        options.setdefault("rho", 500)
        self.rho = options['rho']
        # the ratio of active user every round
        options.setdefault("active_ratio", 1)
        self.active_ratio = options['active_ratio']


class nonsmooth:
    def __init__(self, h):
        if h == "l1":
            self.h = regularizer.l1_value
            self.prox = regularizer.l1_prox
            self.partialprox = regularizer.li_partial_prox

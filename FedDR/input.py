from regularizer import regularizer
import torch.nn as nn


class set_options:
    def __init__(self, **options):
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
        # local epoch
        options.setdefault("local_epoch", 1)
        self.local_epoch = options['local_epoch']
        options.setdefault("learning_rate", 0.1)
        self.learning_rate = options['learning_rate']
        # batch size
        options.setdefault("batch_size", 32)
        self.batch_size = options['batch_size']
        # FedDR
        options.setdefault("eta", 1)
        self.eta = options['eta']
        options.setdefault("alpha", 0.9)
        self.alpha = options['alpha']
        # paritial ratio
        options.setdefault("active_client_ratio", 1)
        self.C = options['active_client_ratio']


class nonsmooth:
    def __init__(self, h):
        if h == "l1":
            self.h = regularizer.l1_value
            self.prox = regularizer.l1_prox
            self.partialprox = regularizer.li_partial_prox

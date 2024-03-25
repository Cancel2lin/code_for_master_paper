import copy
import torch
from torch.utils.data import DataLoader
from utils.deal_dataset import DataSplit, Global_Data


class Client_DR(object):
    def __init__(self, train_dataset, idx, options):
        self.train_data = DataLoader(DataSplit(train_dataset, idx), batch_size=options.batch_size, shuffle=True)
        self.loss_func = options.loss_func
        self.lr = options.learning_rate
        self.epoch = options.local_epoch
        self.alpha = options.alpha
        self.eta = options.eta

    def update_yi(self, global_para, local_xi, local_yi):
        local_yi_new = copy.deepcopy(local_yi)
        for i in global_para.keys():
            local_yi_new[i] = local_yi[i] + self.alpha * (global_para[i] - local_xi[i])

        return local_yi_new

    def update_xi(self, global_para, local_yi, model):
        global_model = copy.deepcopy(model)
        local_model = copy.deepcopy(model)
        global_model.load_state_dict(local_yi)
        local_model.load_state_dict(global_para)
        for epoch in range(self.epoch):
            for _, (batch_x, batch_y) in enumerate(self.train_data):
                batch_output = local_model(batch_x.float())
                batch_loss = self.loss_func(batch_output, batch_y.type(torch.LongTensor))
                batch_loss.backward()
                for w, w_t in zip(local_model.parameters(), global_model.parameters()):
                    grads = w.grad + (w.data - w_t.data) / self.eta
                    w.data = w.data - self.lr * grads
                local_model.zero_grad()
        return local_model.state_dict()

    @staticmethod
    def update_xhat(local_xi, local_yi):
        local_xhat = copy.deepcopy(local_xi)
        for i in local_xi.keys():
            local_xhat[i] = 2 * local_xi[i] - local_yi[i]
        return local_xhat


def evaluate(dataset, global_model, options):
    data = DataLoader(Global_Data(dataset), batch_size=20, shuffle=False)
    loss_func = options.loss_func
    num_sample, loss, correct = 0.0, 0.0, 0.0
    for _, (batch_x, batch_y) in enumerate(data):
        num_sample += len(batch_y)
        batch_output = global_model(batch_x)
        batch_loss = loss_func(batch_output, batch_y.type(torch.LongTensor))
        loss += batch_loss.item() * len(batch_y)

        y_pred = torch.max(batch_output, 1)[1]
        y_pred = y_pred.view(-1)

        correct += torch.sum(torch.eq(y_pred, batch_y)).item()

    loss /= num_sample
    correct /= num_sample

    return loss, correct

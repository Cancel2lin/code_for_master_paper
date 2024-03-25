import torch
from utils.swith_variable_type import mat_switch_vec


# caculate the gradient and hessian matrix
def grad_and_hessian(model, loss_func, train_data, dim, hessian=True):
    if hessian:
        model.zero_grad()
        for _, (batch_x, batch_y) in enumerate(train_data):
            batch_output = model(batch_x.float())
            batch_loss = loss_func(batch_output, batch_y.type(torch.LongTensor))

        # caculate hessian
        grads = torch.autograd.grad(batch_loss, model.parameters(), retain_graph=True, create_graph=True)
        a = []
        for k in range(len(grads)):
            for i in range(grads[k].size(0)):
                if len(grads[k].size()) == 2:     # store about para.weight
                    for j in range(grads[k].size(1)):
                        a.append(torch.autograd.grad(grads[k][i][j], model.parameters(), retain_graph=True,
                                                     create_graph=True))
                else:    # store about para.bias
                    a.append(torch.autograd.grad(grads[k][i], model.parameters(), retain_graph=True,
                                                 create_graph=True))
        # joint the second gradient of every para's and make them to hessian matrix
        c = torch.Tensor([0 for _ in range(dim)])
        for i in range(len(a)):
            b = []
            for j in range(len(a[i])):
                b.append(mat_switch_vec(a[i][j]))
            b = torch.hstack(b)
            c = torch.vstack((c, b))

        hessian_mat = c[1:].detach().numpy()
        hessian_mat = torch.tensor(hessian_mat)

        # caculate gradient
        batch_loss.backward()
        grad = []
        for _, parms in model.named_parameters():
            grad.append(mat_switch_vec(parms.grad))
        grad = torch.hstack(grad)
    else:
        model.zero_grad()
        for _, (batch_x, batch_y) in enumerate(train_data):
            batch_output = model(batch_x.float())
            batch_loss = loss_func(batch_output, batch_y.type(torch.LongTensor))
        # caculate gradient
        batch_loss.backward()
        grad = []
        for _, parms in model.named_parameters():
            grad.append(mat_switch_vec(parms.grad))
        grad = torch.hstack(grad)
        hessian_mat = torch.eye(dim)

    return grad, hessian_mat

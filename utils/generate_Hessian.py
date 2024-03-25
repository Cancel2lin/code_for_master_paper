import torch


def Hessian_SR1(k, para_now, para_old, grad_dif, nu_hat=1e-6, eta=0.9):
    """
    k：当前轮次
    para_now：当前迭代的模型参数,格式：torch.tensor('w','b') 'w'.shape=[1,dimension*n_class]
    para_old：上一轮迭代的模型参数,格式：同上
    grad_dif: 梯度差值
    """
    dim = len(para_now)
    if k == 0:
        B_0 = 1
        u = torch.zeros(dim)
    else:
        s_k = (para_now - para_old).to(torch.float32)
        y_k = grad_dif.to(torch.float32)
        sy = torch.dot(s_k, y_k)
        ss = torch.dot(s_k, s_k)
        if sy >= nu_hat * ss:
            nu_k = 0
        else:
            nu_k = torch.maximum(torch.tensor(0), -sy / ss) + nu_hat
        z_k = y_k + nu_k * s_k
        sz = torch.dot(s_k, z_k)
        zz = torch.dot(z_k, z_k)
        if (ss / sz) ** 2 - ss / zz < 0:
            gamma_k = eta * sz / zz
        else:
            gamma_k = torch.minimum(ss / sz - torch.sqrt((ss / sz) ** 2 - ss / zz), eta * sz / zz)

        B_0 = 1
        u = (gamma_k * z_k - s_k) / torch.sqrt(ss - gamma_k * sz)

    B_k = B_0 * torch.eye(dim) - u * u.unsqueeze(-1)

    return B_k, B_0, u

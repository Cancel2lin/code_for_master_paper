import torch
from utils.subroutines import bisection, semisoomth_newton


def proximal_compute(x_hat, prox, partial_prox, B_0, U, lambd):
    """
    x_hat: inner solution
    prox: the proximal function, determine by the non-smooth regularize
    partial_prox : the Clarke Jacobian of the proximal of the regularize
    B_0, U: the component of Hessian matrix
    options: included the options we need, such as the regularize coefficient(lambd),
    mu: for proximal operator
    global_para: tensor('w','b')
    """
    dim = U.shape[-1]
    # L(a)
    reg_l = lambd
    L_alpha = lambda alpha: L(alpha, x_hat, B_0, U, prox, reg_l, dim)
    partialL_alpha = lambda alpha: partial_L(alpha, x_hat, B_0, U, partial_prox, reg_l)
    L_norm = lambda alpha: 1 / 2 * torch.linalg.norm(L_alpha(alpha)) ** 2
    if dim == 1:
        alpha = bisection(L_alpha, x_hat, U.squeeze(-1))
        x = x_hat + U.squeeze(-1) * alpha
    else:
        alpha = semisoomth_newton(L_alpha, partialL_alpha, L_norm, x0=torch.zeros(dim))
        x = x_hat + torch.matmul(U.to(torch.float32), alpha)

    prox_dir = prox(x, reg_l / B_0)

    return prox_dir


# L(a)=0
def L(alpha, x_hat, B_0, U, prox, reg_l, dim):
    if dim == 1:
        alpha = alpha.unsqueeze(-1)
    else:
        alpha = alpha
    x = x_hat + 1 / B_0 * torch.matmul(U.to(torch.float32), alpha.to(torch.float32))
    prox_x = prox(x, reg_l / B_0)
    L = torch.matmul(U.T.to(torch.float32), (x_hat - prox_x).to(torch.float32)) + alpha

    return L


def partial_L(alpha, x_hat, B_0, U, partial_prox, reg_l):
    dim = U.shape[-1]
    x = x_hat + 1 / B_0 * torch.matmul(U.to(torch.float32), alpha)

    partial = partial_prox(x, reg_l / B_0)
    grad_L = -1 / B_0 * torch.matmul(torch.matmul(U.T.to(torch.float32), partial), U.to(torch.float32)) + torch.eye(dim)

    return grad_L

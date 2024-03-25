import numpy as np
import torch
import torch.nn as nn

from utils.swith_variable_type import vec_swith_mat, dict_swtich_order


def bisection(f, x, u):
    """
    for solving L(a)=<u, x-prox(x+ua)>+a=0 when r=1
    f: f(a), function. In here is L(a)
    """
    K = 100
    eps = 1e-8
    beta = torch.linalg.norm(u) * 2 * torch.linalg.norm(x)
    alpha_minus = -beta
    alpha_plus = beta
    alpha_k = alpha_old = 0
    for k in range(K):
        alpha_k = 1 / 2 * (alpha_minus + alpha_plus)
        if f(alpha_k) > 0:
            alpha_plus = alpha_k
        else:
            alpha_minus = alpha_k
        if k > 1 & (torch.abs(alpha_k - alpha_old) < eps):
            break
        alpha_old = alpha_k
    return alpha_k


def semisoomth_newton(f, grad_f, f_norm, x0, delta=0.5, beta=0.5):
    """
    f：f(x), function
    grad_f: The gradient of f(x), function
    x0：initiation, tensor类型
    """
    K = 300
    eps = 1e-8
    x = x0
    k = 0
    while k <= K and (torch.linalg.norm(f(x)) >= eps):
        k += 1
        f_value = f(x)
        grad = grad_f(x)
        dir = - torch.linalg.solve(grad, f_value)
        t = 1
        while f_norm(x + t * dir) > (1 - 2 * delta * t) * f_norm(x):
            t *= beta

        x += t * dir

    return x


def conjugate_grad_method(A, b, x0=None):
    """
    find the solution of Ax=b.
    x0: init
    """
    eps = 1e-9
    K = 100
    dim = A.shape[0]
    if x0 is None:
        x0 = torch.zeros(dim)
    x = x0
    r = b - torch.matmul(A.to(torch.float32), x.to(torch.float32))
    q = r
    for k in range(K):
        rr = np.dot(r, r)
        qq = np.dot(np.dot(A, q), q)
        if qq == 0:
            return x
        a = rr / qq
        x += a * q
        r -= a * np.dot(A, q)
        if torch.norm(r) < eps:
            break
        beta = np.dot(r, r) / rr
        q = r + beta * q
    if torch.norm(r) >= eps:
        print("The conjugate gradient method don't reach the terminate condition!")
    return x


def line_search(train_data, model, x, loss, h, lambd, grad, dir, dim):
    """
    grad: 光滑项的梯度；tensor类型
    h: 正则项；function
    f: 光滑项在x处的值；tensor类型
    dir: 搜索方向；tensor类型
    dim: n_class*dimension
    """
    K = 10
    delta = 0.5
    beta = 0.8
    t = 1

    h_value = h(x, lambd)
    fun_value = loss + h_value

    test_model = model
    loss_func = nn.CrossEntropyLoss()
    for i in range(K):
        x_test = x + t * dir
        h_plus_value = h(x + dir, lambd)
        lambd = np.dot(grad, dir) + h_plus_value - h_value

        # 将x_test转为orderdict类型
        x_test_weight = x_test[:dim]
        x_test_bias = x_test[dim:]
        x_test_weight = vec_swith_mat(x_test_weight, len(x_test_bias))
        x_test_order = {'predict.weight': x_test_weight, 'predict.bias': x_test_bias}
        x_test_order = dict_swtich_order(x_test_order)

        test_model.load_state_dict(x_test_order)
        test_model.zero_grad()
        num_sample = test_loss = 0.0
        for _, (batch_x, batch_y) in enumerate(train_data):
            num_sample += len(batch_y)
            batch_output = test_model(batch_x.float())
            batch_loss = loss_func(batch_output, batch_y.type(torch.LongTensor))
            test_loss += batch_loss * len(batch_y)
        test_loss /= num_sample
        test_fun_value = test_loss + h(x_test, lambd)

        if test_fun_value <= fun_value + delta * t * lambd:
            break
        t *= beta

    return t

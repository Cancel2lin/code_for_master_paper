import torch
from collections import OrderedDict


# 转换参数数据结构
# 从矩阵转为一列向量 x=(x_1, x_2, ...x_n)',x_i.shape=[d,1] x.shape=[n,d] 转为 x=[x'_1,...,x'_n] x.shape=[1,n*d]
def mat_switch_vec(v):
    """
    v: 矩阵，[m,d]维，将其转为[1,md]维向量
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    para = torch.Tensor([0])
    tmp = v
    if len(tmp.shape) == 1:
        tmp = tmp.unsqueeze(0)
    m = tmp.shape[0]
    if m == 1:
        para = v
    else:
        for i in range(m):
            para = torch.cat((para, v[i]))
        para = para[1:]

    return para


# 将向量转为矩阵
def vec_swith_mat(v, nrow):
    """
    :param v:  向量；nrow：矩阵的第一维度（即行数）
    :return:
    """
    dim_all = len(v)
    ncol = dim_all // nrow

    mat = v.reshape(nrow, ncol)

    return mat


# 将OrderedDict类型转为dict类型
def order_swith_dict(orderdict):
    """
    :param orderdict: OrderedDict数据类型

    :return: v: dict数据类型，即{"w":[],"b":[]}
    """
    v = dict(orderdict)

    return v


# 将dict类型转为OrderedDict类型
def dict_swtich_order(dic):
    v = OrderedDict(dic)

    return v

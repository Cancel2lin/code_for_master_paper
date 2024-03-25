import torch


def l1_value(x, lambd):
    """
    l1_regularizer: lambd * ||x||_1
    """
    return lambd * torch.norm(x, p=1)


def l1_prox(x, lambd):
    return torch.sign(x) * torch.maximum(torch.abs(x) - lambd, torch.tensor(0))


def li_partial_prox(x, lambd):
    partial_diag = (abs(x) > lambd).long()
    partial_prox = torch.diag(partial_diag).to(torch.float32)

    return partial_prox

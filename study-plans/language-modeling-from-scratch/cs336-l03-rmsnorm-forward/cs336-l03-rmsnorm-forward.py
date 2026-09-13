import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """

    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)

    answer = (x / rms) * g

    return answer
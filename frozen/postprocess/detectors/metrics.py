import numpy as np
import torch
import torch.nn.functional as F
MAXIMUM_EFFICIENT_MATRIX_SIZE = 20_000_000_000

def torch_wd(x: torch.Tensor, y: torch.Tensor, k):
    """
    Computes the segmentation error between two sequences x and y using a window size k.
    :param x: The hypothesis sequence.
    :param y: The reference sequence.
    :param k: The segmentation window size.
    :return: A value between 0 and 1 with gradient support.
    """
    kernel = torch.ones(1, 1, k, dtype=x.dtype, device=x.device, requires_grad=False)
    x, y = x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0)
    return torch.clip(torch.abs(F.conv1d(x, kernel).squeeze() - F.conv1d(y, kernel).squeeze()), 0, 1).float().mean()

def torch_pk(x: torch.Tensor, y: torch.Tensor, k):
    """
    Computes the Pk Score between two sequences x and y using a window size k.
    :param x: The hypothesis sequence.
    :param y: The reference sequence.
    :param k: The segmentation window size.
    :return: A value between 0 and 1 with gradient support.
    """
    kernel = torch.ones(1, 1, k, dtype=x.dtype, device=x.device, requires_grad=False)
    x, y = x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0)
    return torch.clip(F.conv1d(torch.abs(x - y), kernel), 0, 1).squeeze().float().mean()


def windowdiff_pk_score(ref, hyp, k=None):
    """
    Computes both WindowDiff and PkScore using efficient 1D convolutions.

    Parameters:
    ref (Tensor): Binary tensor (1 if there is a segmentation boundary, 0 otherwise).
    hyp (Tensor): Binary tensor (1 if there is a segmentation boundary, 0 otherwise).
    k (int): Window size for comparison.

    Returns:
    tuple: (WindowDiff score, PkScore)
    """
    if not isinstance(k, int):
        k = max(min(len(ref) // 10, MAXIMUM_EFFICIENT_MATRIX_SIZE // len(ref)), 3)
    else:
        if k * len(ref) > MAXIMUM_EFFICIENT_MATRIX_SIZE:
            raise ValueError(f"WindowDiff/PkScore: The window size is too large, {k} * {len(ref)} > "
                             f"MAXIMUM_EFFICIENT_MATRIX_SIZE. Operation can overflow memory.")

    ref, hyp = (torch.tensor(ref, dtype=torch.uint8, requires_grad=False),
                torch.tensor(hyp, dtype=torch.uint8, requires_grad=False))
    kernel = torch.ones(1, 1, k, dtype=torch.uint8, requires_grad=False)
    ref, hyp = ref.unsqueeze(0).unsqueeze(0), hyp.unsqueeze(0).unsqueeze(0)

    ref_cuts, hyp_cuts = F.conv1d(ref, kernel).squeeze(), F.conv1d(hyp, kernel).squeeze()
    wd_score = (ref_cuts != hyp_cuts).float().mean()
    pk_score = torch.clip(F.conv1d((ref != hyp).byte(), kernel).squeeze(), 0, 1).float().mean()

    return wd_score.item(), pk_score.item()

def np_wd_pk(seg1, seg2, k=None):
    """
    Computes Pk Score and WindowDiff in an optimized way using NumPy.

    :param seg1: np.array, first segmentation. -> hypothesis (x)
    :param seg2: np.array, second segmentation. -> reference (y)
    :param k: int, window size.
    :return: tuple(WindowDiff score, PkScore)
    """
    if k is None:
        k = len(seg2) // 10

    num_windows = len(seg1) - k + 1

    seg1_windows = np.lib.stride_tricks.sliding_window_view(seg1, k)
    seg2_windows = np.lib.stride_tricks.sliding_window_view(seg2, k)

    pk_count = np.all(seg1_windows != seg2_windows, axis=1).sum()

    wd_count = (np.abs(np.sum(seg1_windows - seg2_windows, axis=1)) > 0).astype(int).sum()

    return wd_count / num_windows, pk_count / num_windows



def wd_pk(seg1, seg2, k=None):
    """
    This is a Python implementation of the Pk Score and WindowDiff metrics.
    :param seg1:
    :param seg2:
    :param k:
    :return:
    """
    if k is None:
        k = len(seg2) // 10
    pk_count = 0
    wd_count = 0
    for i in range(len(seg1) - k + 1):
        seg1_window = seg1[i:i+k]
        seg2_window = seg2[i:i+k]
        pk_count += np.all(seg1_window != seg2_window)
        wd_count += (np.abs(np.sum(seg1_window - seg2_window)) > 0).astype(int)
        if i % 1000 == 0:
            print(f"Processed {i}/{(len(seg1) - k + 1)/1000} windows")
    return wd_count / (len(seg1) - k + 1), pk_count / (len(seg1) - k + 1)

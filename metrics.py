import numpy as np
from scipy.ndimage import morphology


def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.detach().numpy()
    return x

def surfd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)

    # ~s is the negative of s
    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds

def metrics(mask_, gt_, sampling=1):
    lnot = np.logical_not
    land = np.logical_and

    true_positive = np.sum(land((mask_), (gt_)))
    false_positive = np.sum(land((mask_), lnot(gt_)))
    false_negative = np.sum(land(lnot(mask_), (gt_)))
    true_negative = np.sum(land(lnot(mask_), lnot(gt_)))

    M = np.array([[true_negative, false_negative],
                  [false_positive, true_positive]]).astype(np.float64)
    metrics = {}
    metrics['Sensitivity'] = M[1, 1] / (M[0, 1] + M[1, 1] + 1e-5)
    metrics['Specificity'] = M[0, 0] / (M[0, 0] + M[1, 0] + 1e-5)
    metrics['Dice'] = 2 * M[1, 1] / (M[1, 1] * 2 + M[1, 0] + M[0, 1] + 1e-5)
    
    surf_dis = surfd(mask_, gt_, sampling)
    # add hausdorff distance metric
    if len(surf_dis)==0:
        metrics['Hausdorff'] = 0
        metrics['Mean Distance Error'] = 0
    else:
        metrics['Hausdorff'] = surf_dis.max()
        # add mean distance metric
        metrics['Mean Distance Error'] = surf_dis.mean()
    return np.array([metrics['Dice'], metrics['Hausdorff'], metrics['Mean Distance Error']])


def evalAllmetric(mask_, gt_, sampling=1):
    num_region = len(gt_)
    cur_metrics = []
    for num in range(0, int(num_region)):
        cur_metrics.append(metrics(mask_[num], gt_[num], sampling))
    cur_metrics = np.concatenate(cur_metrics)
    return cur_metrics


def det_jac(array):
    # (2,h,w)
    # (x,y)
    dx = np.roll(array, axis = -1, shift = -1) - array
    dy = np.roll(array, axis = -2, shift = -1) - array
    pos_jac = (1+dx[0]) * (1+dy[1]) - dx[1] * dy[0]
    return pos_jac


def gradient_det_jac(array):
    dx = np.roll(array, axis = -1, shift = -1) - array
    dy = np.roll(array, axis = -2, shift = -1) - array
    dis = np.sqrt(dx**2 + dy**2)
    return dis

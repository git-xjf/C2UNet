import math
import numpy as np
import torch

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


EPS = 1e-8


def _extract_logits(output):
    if isinstance(output, dict):
        if 'logits' in output:
            return output['logits']
        if 'final' in output:
            return output['final']
        raise ValueError('Model output dict must contain `logits` or `final`.')
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output



def _to_prob_and_target(output, target):
    output = _extract_logits(output)
    if torch.is_tensor(output):
        prob = torch.sigmoid(output).detach().cpu().numpy()
    else:
        prob = np.asarray(output, dtype=np.float32)
    if torch.is_tensor(target):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = np.asarray(target, dtype=np.float32)
    return prob.astype(np.float32), target_np.astype(np.float32)



def iou_score(output, target, threshold=0.5):
    smooth = 1e-5
    prob, target_np = _to_prob_and_target(output, target)

    output_ = prob > threshold
    target_ = target_np > threshold

    intersection = (output_ & target_).sum(axis=(1, 2, 3))
    union = (output_ | target_).sum(axis=(1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (output_.sum(axis=(1, 2, 3)) + target_.sum(axis=(1, 2, 3)) + smooth)
    return float(iou.mean()), float(dice.mean())



def _mask_diag(mask_shape):
    return float(math.sqrt(sum(s * s for s in mask_shape)))



def _surface_distances(result, reference):
    footprint = generate_binary_structure(result.ndim, 1)
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1, border_value=0)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1, border_value=0)

    if not np.any(result_border):
        result_border = result
    if not np.any(reference_border):
        reference_border = reference

    dt = distance_transform_edt(~reference_border)
    return dt[result_border]



def _symmetric_surface_distances(pred_mask, gt_mask):
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)

    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask[0]
    if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask[0]

    if not SCIPY_AVAILABLE:
        return None

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return np.array([0.0], dtype=np.float64)

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        penalty = _mask_diag(pred_mask.shape)
        return np.array([penalty], dtype=np.float64)

    sd1 = _surface_distances(pred_mask, gt_mask)
    sd2 = _surface_distances(gt_mask, pred_mask)
    return np.concatenate([sd1, sd2]).astype(np.float64)



def binary_hd95_assd(pred_mask, gt_mask):
    all_sd = _symmetric_surface_distances(pred_mask, gt_mask)
    if all_sd is None:
        return float('nan'), float('nan')
    hd95 = float(np.percentile(all_sd, 95))
    assd = float(all_sd.mean())
    return hd95, assd



def binary_bd_bdsd(pred_mask, gt_mask):
    all_sd = _symmetric_surface_distances(pred_mask, gt_mask)
    if all_sd is None:
        return float('nan'), float('nan')
    bd = float(all_sd.mean())
    bdsd = float(all_sd.std())
    return bd, bdsd



def _centroid(gt):
    h, w = gt.shape
    if gt.sum() == 0:
        return w // 2, h // 2
    ys, xs = np.where(gt > 0)
    x = int(np.round(xs.mean()))
    y = int(np.round(ys.mean()))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return x, y



def _ssim(pred, gt):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    n = pred.size
    if n == 0:
        return 0.0

    x = pred.mean()
    y = gt.mean()
    if n == 1:
        sigma_x2 = 0.0
        sigma_y2 = 0.0
        sigma_xy = 0.0
    else:
        sigma_x2 = ((pred - x) ** 2).sum() / (n - 1 + EPS)
        sigma_y2 = ((gt - y) ** 2).sum() / (n - 1 + EPS)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + EPS)

    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        return float(alpha / (beta + EPS))
    if alpha == 0 and beta == 0:
        return 1.0
    return 0.0



def _object_score(pred, gt):
    vals = pred[gt > 0]
    if vals.size == 0:
        return 0.0
    x = vals.mean()
    sigma_x = vals.std()
    return float((2.0 * x) / (x * x + 1.0 + sigma_x + EPS))



def _object_measure(pred, gt):
    u = gt.mean()
    fg_score = _object_score(pred, gt)
    bg_score = _object_score(1.0 - pred, 1.0 - gt)
    return float(u * fg_score + (1.0 - u) * bg_score)



def _region_measure(pred, gt):
    x, y = _centroid(gt)
    h, w = gt.shape

    lt_pred = pred[: y + 1, : x + 1]
    rt_pred = pred[: y + 1, x + 1 :]
    lb_pred = pred[y + 1 :, : x + 1]
    rb_pred = pred[y + 1 :, x + 1 :]

    lt_gt = gt[: y + 1, : x + 1]
    rt_gt = gt[: y + 1, x + 1 :]
    lb_gt = gt[y + 1 :, : x + 1]
    rb_gt = gt[y + 1 :, x + 1 :]

    area = float(h * w)
    weights = [
        ((y + 1) * (x + 1)) / area,
        ((y + 1) * (w - x - 1)) / area,
        ((h - y - 1) * (x + 1)) / area,
        ((h - y - 1) * (w - x - 1)) / area,
    ]

    scores = [
        _ssim(lt_pred, lt_gt),
        _ssim(rt_pred, rt_gt),
        _ssim(lb_pred, lb_gt),
        _ssim(rb_pred, rb_gt),
    ]
    return float(sum(wi * si for wi, si in zip(weights, scores)))



def structure_measure(pred_mask, gt_mask, alpha=0.5):
    pred = np.asarray(pred_mask, dtype=np.float32)
    gt = np.asarray(gt_mask, dtype=np.float32)

    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = gt[0]

    pred = np.clip(pred, 0.0, 1.0)
    gt = (gt > 0.5).astype(np.float32)

    y = float(gt.mean())
    if y == 0.0:
        return float(1.0 - pred.mean())
    if y == 1.0:
        return float(pred.mean())

    so = _object_measure(pred, gt)
    sr = _region_measure(pred, gt)
    score = alpha * so + (1.0 - alpha) * sr
    return float(max(0.0, score))



def compute_seg_metrics(output, target, threshold=0.5):
    prob, target_np = _to_prob_and_target(output, target)

    pred = prob > threshold
    gt = target_np > threshold

    smooth = 1e-5
    intersections = (pred & gt).sum(axis=(1, 2, 3))
    unions = (pred | gt).sum(axis=(1, 2, 3))
    pred_sum = pred.sum(axis=(1, 2, 3))
    gt_sum = gt.sum(axis=(1, 2, 3))

    iou = ((intersections + smooth) / (unions + smooth)).astype(np.float64)
    dice = ((2.0 * intersections + smooth) / (pred_sum + gt_sum + smooth)).astype(np.float64)

    hd95_list = []
    assd_list = []
    bd_list = []
    bdsd_list = []
    salpha_list = []

    for i in range(prob.shape[0]):
        hd95, assd = binary_hd95_assd(pred[i], gt[i])
        bd, bdsd = binary_bd_bdsd(pred[i], gt[i])
        salpha = structure_measure(prob[i], target_np[i], alpha=0.5)
        hd95_list.append(hd95)
        assd_list.append(assd)
        bd_list.append(bd)
        bdsd_list.append(bdsd)
        salpha_list.append(salpha)

    return {
        'iou': float(np.mean(iou)),
        'dice': float(np.mean(dice)),
        'hd95': float(np.nanmean(hd95_list)),
        'assd': float(np.nanmean(assd_list)),
        'bd': float(np.nanmean(bd_list)),
        'bdsd': float(np.nanmean(bdsd_list)),
        's_alpha': float(np.nanmean(salpha_list)),
    }

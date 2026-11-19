import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except Exception:
    distance_transform_edt = None
    SCIPY_AVAILABLE = False

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    lovasz_hinge = None

__all__ = [
    'DiceLoss',
    'BCEDiceLoss',
    'BoundaryLoss',
    'BoundaryBCEDiceLoss',
    'C2UNetLoss',
    'C2SegLoss',
    'LovaszHingeLoss',
]


def _extract_outputs(output):
    if isinstance(output, dict):
        final_logits = output.get('logits', output.get('final', None))
        if final_logits is None:
            raise ValueError('Model output dict must contain `logits` or `final`.')
        stage_logits = list(output.get('stage_logits', output.get('stages', [])))
        flow = output.get('flow', None)
        init_logits = output.get('init_logits', None)
        if init_logits is not None:
            stage_logits = [init_logits] + stage_logits
        return final_logits, stage_logits, flow

    if isinstance(output, (list, tuple)):
        return output[-1], list(output[:-1]), None

    return output, [], None


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        prob = torch.sigmoid(input)
        num = target.size(0)
        prob = prob.view(num, -1)
        target = target.view(num, -1)
        intersection = (prob * target).sum(1)
        dice = (2.0 * intersection + self.smooth) / (prob.sum(1) + target.sum(1) + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        dice = self.dice(input, target)
        return self.bce_weight * bce + dice


class BoundaryLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def _weight_map(self, prob):
        sigma_sq = max(self.sigma ** 2, 1e-12)
        return torch.exp(-((prob - 0.5) ** 2) / (2.0 * sigma_sq))

    def forward(self, input, target):
        prob = torch.sigmoid(input)
        weight_map = self._weight_map(prob)
        bce_map = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        return (weight_map * bce_map).mean()


class BoundaryBCEDiceLoss(nn.Module):
    def __init__(self, boundary_weight=1.0, sigma=0.1, smooth=1e-5):
        super().__init__()
        self.seg = BCEDiceLoss(smooth=smooth)
        self.boundary = BoundaryLoss(sigma=sigma)
        self.boundary_weight = boundary_weight

    def forward(self, input, target):
        return self.seg(input, target) + self.boundary_weight * self.boundary(input, target)


def _sdf_gradient(mask_np):
    if SCIPY_AVAILABLE:
        pos = distance_transform_edt(mask_np > 0.5)
        neg = distance_transform_edt(mask_np <= 0.5)
        sdf = pos - neg
        gy, gx = np.gradient(sdf.astype(np.float32))
    else:
        mask_t = torch.from_numpy(mask_np.astype(np.float32))[None, None]
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        gx = F.conv2d(mask_t, sobel_x, padding=1)[0, 0].numpy()
        gy = F.conv2d(mask_t, sobel_y, padding=1)[0, 0].numpy()
    return gx.astype(np.float32), gy.astype(np.float32)


def _signed_distance_to_tangent_flow(mask_np):
    """Build a tangent contour flow target by rotating signed-distance gradients by 90 degrees."""
    gx, gy = _sdf_gradient(mask_np)
    tx = -gy
    ty = gx
    flow = np.stack([tx, ty], axis=0)
    norm = np.sqrt((flow ** 2).sum(axis=0, keepdims=True) + 1e-8)
    flow = flow / norm
    return flow.astype(np.float32)


@torch.no_grad()
def build_flow_target_from_mask(target):
    if torch.is_tensor(target):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = np.asarray(target)

    if target_np.ndim != 4:
        raise ValueError(f'Expected target with shape [B, C, H, W], got {target_np.shape}')

    batch_flow = []
    for i in range(target_np.shape[0]):
        mask_np = (target_np[i] > 0.5).any(axis=0).astype(np.float32)
        flow_np = _signed_distance_to_tangent_flow(mask_np)
        batch_flow.append(flow_np)

    flow = np.stack(batch_flow, axis=0)
    return torch.from_numpy(flow).to(dtype=target.dtype if torch.is_tensor(target) else torch.float32)


def gradient_x(x):
    return F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode='replicate')


def gradient_y(x):
    return F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')


class C2UNetLoss(nn.Module):
    """
    Paper-aligned training objective:

        total = seg(final)
              + alpha * flow(flow_pred, flow_gt)
              + decomp_weight * L_dec
              + ctx_weight * L_ctx
              + ctx_smooth_weight * Omega(Z)

    Stage supervision and boundary loss are intentionally not used by C2UNetLoss.
    The model can still contain multiple unrolled stages; only the auxiliary stage
    loss term is removed from the objective.
    """

    def __init__(
        self,
        alpha=0.5,
        flow_loss='smoothl1',
        smooth=1e-5,
        decomp_weight=0.05,
        ctx_weight=0.05,
        ctx_smooth_weight=0.01,
    ):
        super().__init__()
        self.alpha = alpha
        self.decomp_weight = decomp_weight
        self.ctx_weight = ctx_weight
        self.ctx_smooth_weight = ctx_smooth_weight
        self.seg_loss = BCEDiceLoss(smooth=smooth)

        flow_loss = flow_loss.lower()
        if flow_loss == 'l1':
            self.flow_criterion = nn.L1Loss()
        elif flow_loss == 'mse':
            self.flow_criterion = nn.MSELoss()
        else:
            self.flow_criterion = nn.SmoothL1Loss(beta=0.1)

    @staticmethod
    def _resize_target(target, pred):
        if pred is None or target.shape[-2:] == pred.shape[-2:]:
            return target
        return F.interpolate(target, size=pred.shape[-2:], mode='nearest')

    @staticmethod
    def _decomposition_loss(feat, masks_lowres, contexts_feat_lowres):
        if feat is None or masks_lowres is None or contexts_feat_lowres is None:
            return None
        terms = []
        for mask, ctx_feat in zip(masks_lowres, contexts_feat_lowres):
            recon = feat * mask + ctx_feat
            terms.append(0.5 * F.mse_loss(recon, feat))
        return torch.stack(terms).mean() if terms else None

    def _context_regularization_loss(self, contexts_lowres, bacr_aux, target=None):
        if contexts_lowres is None or bacr_aux is None:
            return None, None
        ctx_terms = []
        smooth_terms = []
        # contexts_lowres[0] is Z^0, while bacr_aux[k] corresponds to Z^k.
        for k, aux in enumerate(bacr_aux):
            z_k = contexts_lowres[k + 1]

            # Label-guided context prototype consistency:
            # supervise the prototype-derived logits and the context-derived logits
            # with the binary Kvasir mask. This adapts the multi-label Dice idea to
            # single-label/binary polyp segmentation, instead of forcing Z^k and
            # Psi^k to match by feature-space MSE.
            proto_logits = aux.get('proto_logits', None)
            ctx_logits = aux.get('ctx_logits', None)
            if target is not None and proto_logits is not None and ctx_logits is not None:
                target_proto = self._resize_target(target, proto_logits)
                target_ctx = self._resize_target(target, ctx_logits)
                ctx_terms.append(0.5 * (
                    self.seg_loss(proto_logits, target_proto) +
                    self.seg_loss(ctx_logits, target_ctx)
                ))
            else:
                # Backward-compatible fallback for older checkpoints/models.
                psi_k = aux['psi']
                ctx_terms.append(0.5 * F.mse_loss(z_k, psi_k))

            gx = gradient_x(z_k)
            gy = gradient_y(z_k)

            if target is not None:
                # Label-aware smoothness: keep context smooth inside foreground/background,
                # but reduce the penalty on true mask boundaries to avoid over-smoothing
                # irregular polyp contours.
                target_z = self._resize_target(target, z_k)
                edge = (gradient_x(target_z).abs() + gradient_y(target_z).abs()).clamp(0.0, 1.0)
                edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
                weight = 1.0 - edge
                smooth_terms.append(0.5 * ((gx.pow(2) * weight).mean() + (gy.pow(2) * weight).mean()))
            else:
                smooth_terms.append(0.5 * (gx.pow(2).mean() + gy.pow(2).mean()))
        ctx_loss = torch.stack(ctx_terms).mean() if ctx_terms else None
        smooth_loss = torch.stack(smooth_terms).mean() if smooth_terms else None
        return ctx_loss, smooth_loss

    def forward(self, output, target, flow_gt=None, return_breakdown=True):
        final_logits, _, flow_pred = _extract_outputs(output)

        target_final = self._resize_target(target, final_logits)
        seg = self.seg_loss(final_logits, target_final)

        flow = torch.tensor(0.0, device=final_logits.device)
        if flow_pred is not None:
            if flow_gt is None:
                flow_gt = build_flow_target_from_mask(target)
            flow_gt = flow_gt.to(device=flow_pred.device, dtype=flow_pred.dtype)
            if flow_gt.shape[-2:] != flow_pred.shape[-2:]:
                flow_gt = F.interpolate(flow_gt, size=flow_pred.shape[-2:], mode='bilinear', align_corners=False)
                flow_gt = F.normalize(flow_gt, dim=1, eps=1e-6)
            flow = self.flow_criterion(flow_pred, flow_gt)

        decomp = torch.tensor(0.0, device=final_logits.device)
        ctx_reg = torch.tensor(0.0, device=final_logits.device)
        ctx_smooth = torch.tensor(0.0, device=final_logits.device)
        if isinstance(output, dict):
            feat = output.get('feat_lowres', None)
            masks_lowres = output.get('masks_lowres', None)
            contexts_feat_lowres = output.get('contexts_feat_lowres', None)
            contexts_lowres = output.get('contexts_lowres', None)
            bacr_aux = output.get('bacr_aux', None)

            dec_term = self._decomposition_loss(feat, masks_lowres, contexts_feat_lowres)
            if dec_term is not None:
                decomp = dec_term

            ctx_term, smooth_term = self._context_regularization_loss(contexts_lowres, bacr_aux, target=target)
            if ctx_term is not None:
                ctx_reg = ctx_term
            if smooth_term is not None:
                ctx_smooth = smooth_term

        total = (
            seg
            + self.alpha * flow
            + self.decomp_weight * decomp
            + self.ctx_weight * ctx_reg
            + self.ctx_smooth_weight * ctx_smooth
        )

        if not return_breakdown:
            return total

        return {
            'loss': total,
            'seg': seg.detach(),
            'flow': flow.detach(),
            'decomp': decomp.detach(),
            'ctx': ctx_reg.detach(),
            'ctx_smooth': ctx_smooth.detach(),
        }


class C2SegLoss(C2UNetLoss):
    pass


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if lovasz_hinge is None:
            raise ImportError('Lovasz hinge is not available. Please install LovaszSoftmax first.')

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        return lovasz_hinge(input, target, per_image=True)

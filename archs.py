import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['C2UNetResNet50', 'C2UNetRes2Net50', 'C2UNetPVTV2B2', 'C2UNetPVTV2B1', 'C2UNetPVTV2B0', 'ResNet50UNet', 'UNext_L']


_RESNET50_ENCODER_PREFIXES = ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')


def _strip_known_prefixes(key):
    """Normalize checkpoint keys from DataParallel / project checkpoints / torchvision."""
    for prefix in ('module.', 'model.', 'net.', 'backbone.', 'encoder.'):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Unsupported checkpoint type: {type(checkpoint)}')
    for key in ('state_dict', 'model_state_dict', 'model', 'net'):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _load_torchvision_resnet50_state_dict():
    """Load ImageNet-1K ResNet-50 weights, compatible with new and old torchvision APIs."""
    try:
        from torchvision.models import ResNet50_Weights, resnet50
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
    except TypeError:
        from torchvision.models import resnet50
        return resnet50(pretrained=True).state_dict()
    except Exception as exc:
        raise RuntimeError(
            'Failed to load torchvision ImageNet pretrained ResNet50 weights. '
            'If the compute node cannot access the internet, download the torchvision '
            'resnet50 ImageNet weights first and pass them with --pretrained_ckpt /path/to/resnet50.pth; '
            'or disable this with --pretrained False.'
        ) from exc


def load_resnet50_imagenet_encoder_weights(encoder, input_channels=3, pretrained_ckpt=None):
    """Load ImageNet ResNet-50 weights into the custom ResNetEncoder only.

    The custom encoder keeps torchvision-compatible names for conv1/bn1/layer1-4,
    so the ImageNet state dict can be partially loaded while segmentation heads and
    LGCC/C2UNet modules remain randomly initialized.
    """
    if pretrained_ckpt:
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        source_state = _extract_state_dict(checkpoint)
        source = pretrained_ckpt
    else:
        source_state = _load_torchvision_resnet50_state_dict()
        source = 'torchvision ResNet50_Weights.IMAGENET1K_V1'

    encoder_state = encoder.state_dict()
    load_state = {}
    skipped = []

    for key, value in source_state.items():
        key = _strip_known_prefixes(key)
        if not key.startswith(_RESNET50_ENCODER_PREFIXES):
            continue
        if key not in encoder_state:
            continue
        if encoder_state[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(encoder_state[key].shape)))
            continue
        load_state[key] = value

    if input_channels != 3 and 'conv1.weight' in load_state:
        skipped.append(('conv1.weight', tuple(load_state['conv1.weight'].shape), tuple(encoder_state['conv1.weight'].shape)))
        load_state.pop('conv1.weight')
        warnings.warn(
            f'input_channels={input_channels}; skipped ImageNet conv1.weight because it expects 3-channel RGB input.',
            RuntimeWarning,
        )

    if len(load_state) == 0:
        raise RuntimeError(
            f'No compatible ResNet50 encoder weights were found in {source}. '
            'Expected keys such as conv1.weight, bn1.weight, layer1.0.conv1.weight, ...'
        )

    missing, unexpected = encoder.load_state_dict(load_state, strict=False)
    print(f'=> loaded ImageNet pretrained ResNet50 encoder weights from {source} ({len(load_state)} tensors)')
    if skipped:
        print(f'=> skipped {len(skipped)} incompatible pretrained tensors; first skipped: {skipped[0]}')
    if unexpected:
        print(f'=> unexpected keys while loading encoder weights: {unexpected[:5]}')
    # Missing keys are expected only when conv1 is skipped for non-RGB inputs.
    if missing and not (input_channels != 3 and missing == ['conv1.weight']):
        print(f'=> missing encoder keys after pretrained load: {missing[:5]}')


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, input_channels=3, block=Bottleneck, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


class TimmEncoder(nn.Module):
    """Feature-only timm encoder wrapper for Res2Net/PVT-style backbones.

    It returns the same tuple format as ResNetEncoder: (x0, x1, x2, x3, x4).
    x0 is set to None because C2UNet only consumes x1-x4.
    """

    def __init__(
        self,
        model_name,
        input_channels=3,
        pretrained=False,
        pretrained_ckpt=None,
        out_indices=(1, 2, 3, 4),
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                'Res2Net/PVT backbones require timm. Install it with: pip install timm'
            ) from exc

        create_kwargs = dict(
            features_only=True,
            out_indices=tuple(out_indices),
            in_chans=input_channels,
        )
        # Offline-safe loading:
        # - If pretrained_ckpt is provided, load the ImageNet checkpoint from that local file.
        #   Using pretrained_cfg_overlay={'file': ...} makes timm treat it as the pretrained
        #   source and avoids any Hugging Face / internet request.
        # - If pretrained=True and no local file is provided, timm will try its normal online/cache
        #   mechanism, which may fail on no-network compute nodes.
        if pretrained_ckpt:
            create_kwargs['pretrained_cfg_overlay'] = {'file': pretrained_ckpt}
            pretrained_flag = True
            source = f'local timm checkpoint: {pretrained_ckpt}'
        else:
            pretrained_flag = bool(pretrained)
            source = 'timm pretrained weights from cache/hub' if pretrained_flag else 'random initialization'

        try:
            self.encoder = timm.create_model(model_name, pretrained=pretrained_flag, **create_kwargs)
        except Exception as exc:
            if pretrained_flag and not pretrained_ckpt:
                raise RuntimeError(
                    f'Failed to build pretrained timm encoder {model_name}. '
                    'This usually means the node has no internet and the weights are not in the local cache. '
                    'Download the corresponding timm ImageNet checkpoint on a machine with internet, copy it to this server, '
                    'and run with --pretrained_ckpt /path/to/pytorch_model.bin. '
                    'For a structure-only run, use --pretrained False.'
                ) from exc
            raise
        self.model_name = model_name
        self.out_indices = tuple(out_indices)
        self.out_channels = tuple(int(c) for c in self.encoder.feature_info.channels())
        if len(self.out_channels) != 4:
            raise RuntimeError(
                f'{model_name} with out_indices={self.out_indices} returned {len(self.out_channels)} feature levels; expected 4.'
            )
        print(f'=> built timm encoder {model_name} from {source}; feature channels={self.out_channels}')

    def forward(self, x):
        feats = self.encoder(x)
        if len(feats) != 4:
            raise RuntimeError(f'{self.model_name} returned {len(feats)} feature maps; expected 4.')
        return None, feats[0], feats[1], feats[2], feats[3]


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FeatureFusionNeck(nn.Module):
    def __init__(self, out_channels=256, in_channels=(256, 512, 1024, 2048)):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f'FeatureFusionNeck expects four encoder feature levels, got {len(in_channels)}')
        self.in_channels = tuple(int(c) for c in in_channels)
        self.lat1 = conv1x1(self.in_channels[0], out_channels)
        self.lat2 = conv1x1(self.in_channels[1], out_channels)
        self.lat3 = conv1x1(self.in_channels[2], out_channels)
        self.lat4 = conv1x1(self.in_channels[3], out_channels)

        self.fuse = nn.Sequential(
            ConvBNReLU(out_channels * 4, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x1, x2, x3, x4):
        size = x1.shape[-2:]
        f1 = self.lat1(x1)
        f2 = F.interpolate(self.lat2(x2), size=size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.lat3(x3), size=size, mode='bilinear', align_corners=False)
        f4 = F.interpolate(self.lat4(x4), size=size, mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([f1, f2, f3, f4], dim=1))


class FlowHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(feat_dim, feat_dim),
            nn.Conv2d(feat_dim, 2, kernel_size=1, bias=True),
        )

    def forward(self, feat):
        flow = self.head(feat)
        flow = F.normalize(flow, dim=1, eps=1e-6)
        return flow


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, hidden_channels=128):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class ContextInitializer(nn.Module):
    def __init__(self, feat_dim, ctx_dim):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(feat_dim * 2, ctx_dim),
            ConvBNReLU(ctx_dim, ctx_dim),
        )

    def forward(self, feat, mask):
        return self.block(torch.cat([feat, feat * (1.0 - mask)], dim=1))


def gradient_x(x):
    return F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode='replicate')


def gradient_y(x):
    return F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')


def divergence(field):
    fx = field[:, 0:1]
    fy = field[:, 1:2]
    dfx = fx - F.pad(fx[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')
    dfy = fy - F.pad(fy[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')
    return dfx + dfy


def _positive_raw(init_value):
    """Return an inverse-softplus scalar so softplus(raw) ~= init_value."""
    value = torch.tensor(float(init_value)).clamp_min(1e-6)
    return torch.log(torch.expm1(value))


class CGMU(nn.Module):
    """Linearized ADMM mask update used by the tutor-proposed Method 2.

    This module keeps the old class name so train.py/cv_search.py and existing
    architecture names remain compatible, but the update is no longer the
    Method-1 entropy/sigmoid update.  For stage k it implements

        V = W * (M - M_hat)

    with a linearized M-subproblem, shrinkage update for V, and multiplier
    update for Lambda.  The proximal map prox_{mu psi / eta} is represented by a
    small learnable residual network, following the model-driven-unrolling style
    already used in the original code.
    """

    def __init__(
        self,
        feat_dim,
        ctx_dim,
        num_classes=1,
        hidden_dim=128,
        eps=1.0,
        lambda1=0.5,
        lambda2=0.25,
        sigma=0.15,
        eta_init=1.0,
        beta_init=0.5,
    ):
        super().__init__()
        self.eps = eps
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sigma = sigma
        self.eta_raw = nn.Parameter(_positive_raw(eta_init))
        self.beta_raw = nn.Parameter(_positive_raw(beta_init))

        # Inputs to the learned proximal map:
        # F, Z, G, M^k, M_admm, grad_decomp, contour residual, div(G), W,
        # V^k, Lambda^k, and the linearized gradient h_tilde.
        prox_in_channels = feat_dim + ctx_dim + 2 + 9 * num_classes
        self.prox = nn.Sequential(
            ConvBNReLU(prox_in_channels, hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )

    @staticmethod
    def _safe_logit(x, eps=1e-4):
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))

    @staticmethod
    def _shrink(x, threshold):
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

    @property
    def eta(self):
        return F.softplus(self.eta_raw) + 1e-6

    @property
    def beta(self):
        return F.softplus(self.beta_raw) + 1e-6

    def _uncertainty_weight(self, prob):
        sigma_sq = max(self.sigma ** 2, 1e-12)
        return torch.exp(-((prob - 0.5) ** 2) / (2.0 * sigma_sq))

    def forward(self, feat, ctx_prev, ctx_feat_prev, flow, lambda_prev, v_prev, mask_prev):
        # W^{k-1} and M_hat^{k-1} are frozen references built from the previous
        # stage prediction, as in Method 2.
        weight = self._uncertainty_weight(mask_prev)
        mask_hat = (mask_prev > 0.5).float()

        # Gradient of 1/2 ||F - F*M - P(Z)||^2 with respect to the scalar mask.
        # The channel mean keeps the update scale stable for large feat_dim.
        recon_residual = feat - feat * mask_prev - ctx_feat_prev
        grad_decomp = -(feat * recon_residual).mean(dim=1, keepdim=True)

        # Contour-consistency term from the tutor method:
        # -lambda1 * <nabla M, G> * div(G).
        grad_mx = gradient_x(mask_prev)
        grad_my = gradient_y(mask_prev)
        contour_residual = grad_mx * flow[:, 0:1] + grad_my * flow[:, 1:2]
        div_g = divergence(flow)
        grad_contour = -self.lambda1 * contour_residual * div_g

        sparse_residual_prev = weight * (mask_prev - mask_hat) - v_prev
        grad_lagrange = weight * lambda_prev
        grad_quadratic = self.beta * weight * sparse_residual_prev
        h_tilde = grad_decomp + grad_contour + grad_lagrange + grad_quadratic

        # Linearized M step followed by a learnable proximal residual.
        mask_admm = (mask_prev - h_tilde / self.eta).clamp(0.0, 1.0)
        prox_input = torch.cat([
            feat,
            ctx_prev,
            flow,
            mask_prev,
            mask_admm,
            grad_decomp,
            contour_residual,
            div_g,
            weight,
            v_prev,
            lambda_prev,
            h_tilde,
        ], dim=1)
        delta = self.prox(prox_input)
        refined_logits = self._safe_logit(mask_admm) + delta
        mask = torch.sigmoid(refined_logits)

        # V and multiplier/Lambda updates.
        shrink_input = weight * (mask - mask_hat) + lambda_prev / self.beta
        v = self._shrink(shrink_input, self.lambda2 / self.beta)
        sparse_residual = weight * (mask - mask_hat) - v
        lambda_next = lambda_prev + self.beta * sparse_residual

        aux = {
            'mask_hat': mask_hat,
            'uncertainty': weight,
            'mask_admm': mask_admm,
            'grad_decomp': grad_decomp,
            'contour_residual': contour_residual,
            'div_g': div_g,
            'h_tilde': h_tilde,
            'sparse_residual_prev': sparse_residual_prev,
            'sparse_residual': sparse_residual,
            'v': v,
            'lambda': lambda_next,
            'eta': self.eta.detach(),
            'beta': self.beta.detach(),
        }
        return refined_logits, mask, v, lambda_next.clamp(-10.0, 10.0), aux



def _import_dcnv4_class():
    """Import the official OpenGVLab DCNv4 op from common installation layouts."""
    import_errors = []
    candidates = (
        'DCNv4_op.DCNv4.modules.dcnv4',
        'DCNv4.modules.dcnv4',
        'dcnv4',
    )
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=['DCNv4'])
            return getattr(module, 'DCNv4')
        except Exception as exc:  # keep all errors for a useful message
            import_errors.append(f'{module_name}: {exc}')
    raise ImportError(
        'BACR was configured with --bacr_prox bc_dcnv4, but the official DCNv4 '
        'operator could not be imported. Install OpenGVLab/DCNv4 first, e.g. clone '
        'https://github.com/OpenGVLab/DCNv4, run `cd DCNv4_op && python setup.py build install`, '
        'or make the repository importable with PYTHONPATH. Import attempts: '
        + ' | '.join(import_errors)
    )


class CNNContextProx(nn.Module):
    """Original fixed-grid CNN proximal map for BACR, wrapped for ablation."""

    def __init__(self, feat_dim, ctx_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(ctx_dim + feat_dim + 1, hidden_dim),
            ConvBNReLU(hidden_dim, ctx_dim),
        )

    def forward(self, ctx_tilde, feat, mask):
        ctx = self.net(torch.cat([ctx_tilde, feat, mask], dim=1))
        return ctx, {}


class DCNv4NCHW(nn.Module):
    """Thin adapter: NCHW tensors -> official DCNv4 (N, H*W, C) -> NCHW."""

    def __init__(
        self,
        channels,
        group=4,
        offset_scale=0.5,
        dw_kernel_size=3,
        center_feature_scale=True,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'DCNv4 channels must be divisible by group, got channels={channels}, group={group}')
        if (channels // group) % 16 != 0:
            raise ValueError(
                'The official DCNv4 CUDA implementation requires channels/group to be divisible by 16. '
                f'Got channels={channels}, group={group}, channels/group={channels // group}. '
                'For hidden_dim=128, use --dcnv4_group 4 or 8.'
            )
        DCNv4 = _import_dcnv4_class()
        self.op = DCNv4(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=group,
            offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size if int(dw_kernel_size) > 0 else None,
            center_feature_scale=bool(center_feature_scale),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        y_seq = self.op(x_seq, shape=(h, w))
        return y_seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BoundaryConditionedDCNv4Prox(nn.Module):
    """Boundary-conditioned DCNv4 proximal refinement for BACR.

    The analytical BACR update still builds ctx_tilde from F*(1-M) and Z^k.
    This module only replaces the final learnable proximal map.  It exposes the
    mask-derived boundary B and uncertainty U to DCNv4, and gates the deformable
    branch so adaptive sampling mainly acts near ambiguous boundaries.
    """

    def __init__(
        self,
        feat_dim,
        ctx_dim,
        hidden_dim=128,
        sigma=0.15,
        dcnv4_group=4,
        dcnv4_offset_scale=0.5,
        dcnv4_dw_kernel_size=3,
        dcnv4_center_feature_scale=True,
        use_boundary_gate=True,
        gate_init=-2.0,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.use_boundary_gate = bool(use_boundary_gate)
        in_ch = ctx_dim + feat_dim + 1 + 1 + 1  # ctx_tilde, feat, mask, boundary, uncertainty
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.dcn = DCNv4NCHW(
            channels=hidden_dim,
            group=dcnv4_group,
            offset_scale=dcnv4_offset_scale,
            dw_kernel_size=dcnv4_dw_kernel_size,
            center_feature_scale=dcnv4_center_feature_scale,
        )
        if self.use_boundary_gate:
            self.boundary_gate = nn.Sequential(
                nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid(),
            )
            # Start close to the original fixed-grid proximal map and let the
            # deformable branch grow only where the loss finds it useful.
            nn.init.constant_(self.boundary_gate[0].weight, 0.0)
            nn.init.constant_(self.boundary_gate[0].bias, float(gate_init))
        else:
            self.boundary_gate = None
        self.out_proj = nn.Sequential(
            ConvBNReLU(hidden_dim, ctx_dim),
        )

    def _uncertainty_weight(self, mask):
        sigma_sq = max(self.sigma ** 2, 1e-12)
        return torch.exp(-((mask - 0.5) ** 2) / (2.0 * sigma_sq))

    @staticmethod
    def _boundary_from_mask(mask):
        boundary = torch.abs(gradient_x(mask)) + torch.abs(gradient_y(mask))
        boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)
        return boundary.clamp(0.0, 1.0)

    def forward(self, ctx_tilde, feat, mask):
        boundary = self._boundary_from_mask(mask)
        uncertainty = self._uncertainty_weight(mask)
        x = torch.cat([ctx_tilde, feat, mask, boundary, uncertainty], dim=1)
        x = self.in_proj(x)
        x_dcn = self.dcn(x)
        if self.boundary_gate is not None:
            gate = self.boundary_gate(torch.cat([mask, boundary, uncertainty], dim=1))
            x = x + gate * (x_dcn - x)
        else:
            gate = None
            x = x_dcn
        ctx = self.out_proj(x)
        aux = {
            'boundary': boundary,
            'uncertainty_ctx': uncertainty,
        }
        if gate is not None:
            aux['boundary_gate'] = gate
        return ctx, aux


class HybridBoundaryConditionedDCNv4Prox(nn.Module):
    """Conservative BACR proximal map: original CNN path plus a small BC-DCNv4 residual.

    Instead of replacing the fixed-grid proximal map, this module preserves the
    original CNN BACR output as the main branch and only applies DCNv4 as a
    boundary-conditioned residual correction:

        ctx = ctx_cnn + alpha * (ctx_dcn - ctx_cnn)

    The residual scale alpha is learnable and initialized small, which makes the
    model start close to the original stable BACR and learn deformable sampling
    only when it consistently improves the objective.
    """

    def __init__(
        self,
        feat_dim,
        ctx_dim,
        hidden_dim=128,
        sigma=0.15,
        dcnv4_group=4,
        dcnv4_offset_scale=0.25,
        dcnv4_dw_kernel_size=3,
        dcnv4_center_feature_scale=True,
        use_boundary_gate=True,
        res_scale_init=0.05,
    ):
        super().__init__()
        self.cnn = CNNContextProx(feat_dim=feat_dim, ctx_dim=ctx_dim, hidden_dim=hidden_dim)
        self.dcn = BoundaryConditionedDCNv4Prox(
            feat_dim=feat_dim,
            ctx_dim=ctx_dim,
            hidden_dim=hidden_dim,
            sigma=sigma,
            dcnv4_group=dcnv4_group,
            dcnv4_offset_scale=dcnv4_offset_scale,
            dcnv4_dw_kernel_size=dcnv4_dw_kernel_size,
            dcnv4_center_feature_scale=dcnv4_center_feature_scale,
            use_boundary_gate=use_boundary_gate,
        )
        init = min(max(float(res_scale_init), 1e-4), 1.0 - 1e-4)
        self.res_scale_raw = nn.Parameter(torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32))

    @property
    def residual_scale(self):
        return torch.sigmoid(self.res_scale_raw)

    def forward(self, ctx_tilde, feat, mask):
        ctx_cnn, aux_cnn = self.cnn(ctx_tilde, feat, mask)
        ctx_dcn, aux_dcn = self.dcn(ctx_tilde, feat, mask)
        alpha = self.residual_scale
        ctx = ctx_cnn + alpha * (ctx_dcn - ctx_cnn)
        aux = {}
        aux.update(aux_cnn)
        aux.update(aux_dcn)
        aux['dcn_residual_scale'] = alpha.detach()
        return ctx, aux


class BACR(nn.Module):
    """Context update aligned with Method 2's Z-subproblem.

    Method 2 writes Z^{k+1} = prox_{nu phi}(F - F*M^{k+1}).  Because the code
    stores Z in ctx_dim channels and reconstructs the feature residual through a
    1x1 projection P(Z), this block first embeds the residual feature
    F - F*M = F*(1-M), blends it with the previous context state as in the
    original unrolling implementation, and then applies a learnable proximal
    refinement.
    """

    def __init__(
        self,
        feat_dim,
        ctx_dim,
        hidden_dim=128,
        num_classes=1,
        lambda3=1.0,
        gamma_init=1.0,
        sigma=0.15,
        prox_type='cnn',
        dcnv4_group=4,
        dcnv4_offset_scale=0.5,
        dcnv4_dw_kernel_size=3,
        dcnv4_center_feature_scale=True,
        use_boundary_gate=True,
        dcnv4_res_scale_init=0.05,
    ):
        super().__init__()
        self.lambda3 = lambda3
        self.prox_type = prox_type
        self.gamma_raw = nn.Parameter(_positive_raw(gamma_init))
        self.residual_embed = nn.Sequential(
            ConvBNReLU(feat_dim, ctx_dim),
            ConvBNReLU(ctx_dim, ctx_dim),
        )
        if prox_type == 'cnn':
            self.refine = CNNContextProx(feat_dim=feat_dim, ctx_dim=ctx_dim, hidden_dim=hidden_dim)
        elif prox_type == 'bc_dcnv4':
            self.refine = BoundaryConditionedDCNv4Prox(
                feat_dim=feat_dim,
                ctx_dim=ctx_dim,
                hidden_dim=hidden_dim,
                sigma=sigma,
                dcnv4_group=dcnv4_group,
                dcnv4_offset_scale=dcnv4_offset_scale,
                dcnv4_dw_kernel_size=dcnv4_dw_kernel_size,
                dcnv4_center_feature_scale=dcnv4_center_feature_scale,
                use_boundary_gate=use_boundary_gate,
            )
        elif prox_type == 'hybrid_bc_dcnv4':
            self.refine = HybridBoundaryConditionedDCNv4Prox(
                feat_dim=feat_dim,
                ctx_dim=ctx_dim,
                hidden_dim=hidden_dim,
                sigma=sigma,
                dcnv4_group=dcnv4_group,
                dcnv4_offset_scale=dcnv4_offset_scale,
                dcnv4_dw_kernel_size=dcnv4_dw_kernel_size,
                dcnv4_center_feature_scale=dcnv4_center_feature_scale,
                use_boundary_gate=use_boundary_gate,
                res_scale_init=dcnv4_res_scale_init,
            )
        else:
            raise ValueError(f'Unknown BACR prox_type: {prox_type}. Use cnn, bc_dcnv4, or hybrid_bc_dcnv4.')

        # Label-aware auxiliary heads for context supervision.  They are used
        # only in C2UNetLoss and are not final segmentation outputs.
        self.proto_label_head = nn.Sequential(
            ConvBNReLU(ctx_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )
        self.ctx_label_head = nn.Sequential(
            ConvBNReLU(ctx_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )

    @property
    def gamma(self):
        return F.softplus(self.gamma_raw) + 1e-6

    def forward(self, feat, mask, ctx_prev):
        residual_feat = feat * (1.0 - mask)
        psi = self.residual_embed(residual_feat)
        ctx_tilde = (self.lambda3 * psi + self.gamma * ctx_prev) / (self.lambda3 + self.gamma)
        ctx, refine_aux = self.refine(ctx_tilde, feat, mask)

        proto_logits = self.proto_label_head(psi)
        ctx_logits = self.ctx_label_head(ctx)

        aux = {
            'psi': psi,
            'ctx_tilde': ctx_tilde,
            'gamma': self.gamma.detach(),
            'proto_logits': proto_logits,
            'ctx_logits': ctx_logits,
            'residual_feat': residual_feat,
            'bacr_prox_type': self.prox_type,
        }
        aux.update(refine_aux)
        return ctx, aux


class C2UNetResNet50(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        deep_supervision=True,
        num_stages=3,
        feat_dim=256,
        ctx_dim=128,
        hidden_dim=128,
        tau=0.5,
        beta=None,
        eps=1.0,
        lambda1=0.5,
        lambda2=0.25,
        lambda3=1.0,
        sigma=0.15,
        eta_init=1.0,
        gamma_init=1.0,
        bacr_prox='cnn',
        dcnv4_group=4,
        dcnv4_offset_scale=0.5,
        dcnv4_dw_kernel_size=3,
        dcnv4_center_feature_scale=True,
        bacr_boundary_gate=True,
        bacr_dcn_start_stage=1,
        bacr_dcn_res_scale_init=0.05,
        pretrained=False,
        pretrained_ckpt=None,
        encoder_name=None,
        timm_out_indices=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.num_stages = num_stages
        self.pretrained = pretrained
        beta_init = tau if beta is None else beta

        if encoder_name is None:
            self.encoder = ResNetEncoder(input_channels=input_channels, layers=(3, 4, 6, 3))
            if self.pretrained:
                load_resnet50_imagenet_encoder_weights(
                    self.encoder,
                    input_channels=input_channels,
                    pretrained_ckpt=pretrained_ckpt,
                )
            neck_in_channels = (256, 512, 1024, 2048)
        else:
            if timm_out_indices is None:
                timm_out_indices = (1, 2, 3, 4)
            self.encoder = TimmEncoder(
                model_name=encoder_name,
                input_channels=input_channels,
                pretrained=pretrained,
                pretrained_ckpt=pretrained_ckpt,
                out_indices=timm_out_indices,
            )
            neck_in_channels = self.encoder.out_channels
        self.neck = FeatureFusionNeck(out_channels=feat_dim, in_channels=neck_in_channels)

        self.init_seg_head = SegmentationHead(feat_dim, num_classes=num_classes, hidden_channels=hidden_dim)
        self.flow_head = FlowHead(feat_dim)
        self.ctx_init = ContextInitializer(feat_dim, ctx_dim)
        self.ctx_to_feat = nn.Conv2d(ctx_dim, feat_dim, kernel_size=1, bias=False)

        self.cgmus = nn.ModuleList([
            CGMU(
                feat_dim=feat_dim,
                ctx_dim=ctx_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                eps=eps,
                lambda1=lambda1,
                lambda2=lambda2,
                sigma=sigma,
                eta_init=eta_init,
                beta_init=beta_init,
            )
            for _ in range(num_stages)
        ])
        dcn_start = int(bacr_dcn_start_stage)
        self.bacrs = nn.ModuleList()
        for stage_idx in range(num_stages):
            stage_no = stage_idx + 1
            stage_bacr_prox = bacr_prox
            if ('dcnv4' in str(bacr_prox)) and stage_no < dcn_start:
                # Early masks are noisy; keep the original stable CNN prox before
                # enabling deformable context refinement in later unrolling stages.
                stage_bacr_prox = 'cnn'
            self.bacrs.append(BACR(
                feat_dim=feat_dim,
                ctx_dim=ctx_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                lambda3=lambda3,
                gamma_init=gamma_init,
                sigma=sigma,
                prox_type=stage_bacr_prox,
                dcnv4_group=dcnv4_group,
                dcnv4_offset_scale=dcnv4_offset_scale,
                dcnv4_dw_kernel_size=dcnv4_dw_kernel_size,
                dcnv4_center_feature_scale=dcnv4_center_feature_scale,
                use_boundary_gate=bacr_boundary_gate,
                dcnv4_res_scale_init=bacr_dcn_res_scale_init,
            ))

    def tau(self, k):
        # Backward-compatible accessor: in Method 2 the old tau argument is
        # used as the ADMM penalty beta when --beta is not explicitly supplied.
        return self.cgmus[k].beta

    def forward(self, x):
        input_size = x.shape[-2:]
        _, x1, x2, x3, x4 = self.encoder(x)
        feat = self.neck(x1, x2, x3, x4)

        init_logits_low = self.init_seg_head(feat)
        mask = torch.sigmoid(init_logits_low)
        flow_low = self.flow_head(feat)
        ctx = self.ctx_init(feat, mask)
        ctx_feat = self.ctx_to_feat(ctx)
        lambda_dual = torch.zeros_like(mask)
        sparse_v = torch.zeros_like(mask)

        stage_logits_low = []
        stage_masks_low = [mask]
        context_states_low = [ctx]
        context_proj_low = [ctx_feat]
        dual_states_low = [lambda_dual]
        sparse_states_low = [sparse_v]
        cgmu_aux = []
        bacr_aux = []
        inconsistency_list = []
        tau_list = []
        beta_list = []

        for k in range(self.num_stages):
            logits_k, mask, sparse_v, lambda_dual, aux_mask = self.cgmus[k](
                feat, ctx, ctx_feat, flow_low, lambda_dual, sparse_v, mask
            )
            ctx, aux_ctx = self.bacrs[k](feat, mask, ctx)
            ctx_feat = self.ctx_to_feat(ctx)

            stage_logits_low.append(logits_k)
            stage_masks_low.append(mask)
            context_states_low.append(ctx)
            context_proj_low.append(ctx_feat)
            dual_states_low.append(lambda_dual)
            sparse_states_low.append(sparse_v)
            inconsistency_list.append(aux_mask['contour_residual'])
            cgmu_aux.append(aux_mask)
            bacr_aux.append(aux_ctx)
            tau_list.append(self.tau(k).detach())
            beta_list.append(self.cgmus[k].beta.detach())

        final_logits_low = stage_logits_low[-1] if len(stage_logits_low) > 0 else init_logits_low
        final_logits = F.interpolate(final_logits_low, size=input_size, mode='bilinear', align_corners=False)
        init_logits = F.interpolate(init_logits_low, size=input_size, mode='bilinear', align_corners=False)
        stage_logits = [
            F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
            for logits in stage_logits_low
        ]
        masks = [
            F.interpolate(mask_k, size=input_size, mode='bilinear', align_corners=False)
            for mask_k in stage_masks_low
        ]
        flow = F.interpolate(flow_low, size=input_size, mode='bilinear', align_corners=False)
        flow = F.normalize(flow, dim=1, eps=1e-6)

        return {
            'logits': final_logits,
            'init_logits': init_logits,
            'stage_logits': stage_logits,
            'flow': flow,
            'flow_lowres': flow_low,
            'feat_lowres': feat,
            'masks': masks,
            'masks_lowres': stage_masks_low,
            'contexts_lowres': context_states_low,
            'contexts_feat_lowres': context_proj_low,
            'dual_lowres': dual_states_low,
            'sparse_v_lowres': sparse_states_low,
            'cgmu_aux': cgmu_aux,
            'bacr_aux': bacr_aux,
            'inconsistency_lowres': inconsistency_list,
            'tau_values': tau_list,
            'beta_values': beta_list,
        }


class C2UNetRes2Net50(C2UNetResNet50):
    def __init__(self, *args, **kwargs):
        # Res2Net has a ResNet-like feature pyramid. Use stages C1-C4: H/4, H/8, H/16, H/32.
        super().__init__(
            *args,
            encoder_name='res2net50_26w_4s',
            timm_out_indices=(1, 2, 3, 4),
            **kwargs,
        )


class C2UNetPVTV2B2(C2UNetResNet50):
    def __init__(self, *args, **kwargs):
        # PVT v2 exposes four pyramid stages directly: H/4, H/8, H/16, H/32.
        super().__init__(
            *args,
            encoder_name='pvt_v2_b2',
            timm_out_indices=(0, 1, 2, 3),
            **kwargs,
        )


class C2UNetPVTV2B1(C2UNetResNet50):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            encoder_name='pvt_v2_b1',
            timm_out_indices=(0, 1, 2, 3),
            **kwargs,
        )


class C2UNetPVTV2B0(C2UNetResNet50):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            encoder_name='pvt_v2_b0',
            timm_out_indices=(0, 1, 2, 3),
            **kwargs,
        )


ResNet50UNet = C2UNetResNet50
UNext_L = C2UNetResNet50

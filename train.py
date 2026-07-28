import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations import Compose, HorizontalFlip, Normalize, RandomRotate90, Resize, VerticalFlip
from torch.optim import lr_scheduler
from tqdm import tqdm
import archs
import losses
from data_utils import build_splits, describe_split_source, find_existing_ext, list_image_ids, resolve_dataset_dir
from dataset import Dataset
from metrics import compute_seg_metrics
from utils import AverageMeter, count_params, str2bool


ARCH_NAMES = sorted(getattr(archs, '__all__', [name for name, obj in archs.__dict__.items() if isinstance(obj, type)]))
LOSS_NAMES = list(losses.__all__) + ['BCEWithLogitsLoss']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: default is dataset_arch')
    parser.add_argument('--epochs', default=400, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='C2UNetResNet50', choices=ARCH_NAMES)
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=512, type=int)
    parser.add_argument('--input_h', default=512, type=int)
    parser.add_argument('--num_stages', default=3, type=int)
    parser.add_argument('--feat_dim', default=256, type=int)
    parser.add_argument('--ctx_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--tau', default=0.5, type=float,
                        help='backward-compatible alias for Method-2 ADMM beta when --beta is not set')
    parser.add_argument('--beta', default=None, type=float,
                        help='ADMM penalty beta for Method-2 V/Lambda updates; default uses --tau')
    parser.add_argument('--eps', default=1.0, type=float)
    parser.add_argument('--lambda1', default=0.5, type=float)
    parser.add_argument('--lambda2', default=0.25, type=float)
    parser.add_argument('--lambda3', default=1.0, type=float)
    parser.add_argument('--sigma', default=0.15, type=float)
    parser.add_argument('--eta_init', default=1.0, type=float)
    parser.add_argument('--gamma_init', default=1.0, type=float)
    parser.add_argument('--bacr_prox', default='cnn', choices=['cnn', 'bc_dcnv4', 'hybrid_bc_dcnv4'],
                        help='BACR proximal map: cnn, direct BC-DCNv4, or conservative CNN+BC-DCNv4 residual')
    parser.add_argument('--dcnv4_group', default=4, type=int,
                        help='DCNv4 groups for BC-DCNv4 BACR; hidden_dim/group must be divisible by 16')
    parser.add_argument('--dcnv4_offset_scale', default=0.5, type=float,
                        help='offset_scale passed to the official DCNv4 operator')
    parser.add_argument('--dcnv4_dw_kernel_size', default=3, type=int,
                        help='depthwise kernel used in DCNv4 offset/mask branch; set 0 to disable')
    parser.add_argument('--dcnv4_center_feature_scale', default=True, type=str2bool,
                        help='enable DCNv4 center_feature_scale')
    parser.add_argument('--bacr_boundary_gate', default=True, type=str2bool,
                        help='gate the DCNv4 branch with mask boundary and uncertainty priors')
    parser.add_argument('--bacr_dcn_start_stage', default=1, type=int,
                        help='1-indexed stage to enable DCNv4 BACR; earlier stages use original CNN BACR')
    parser.add_argument('--bacr_dcn_res_scale_init', default=0.05, type=float,
                        help='initial residual scale for hybrid_bc_dcnv4; smaller is more conservative')
    parser.add_argument('--pretrained', default=True, type=str2bool,
                        help='load ImageNet pretrained encoder weights; ResNet50 uses torchvision, Res2Net/PVT use timm')
    parser.add_argument('--pretrained_ckpt', default=None, type=str,
                        help='optional local ImageNet checkpoint, useful on offline nodes; for timm backbones it is passed as checkpoint_path')

    # loss
    parser.add_argument('--loss', default='C2UNetLoss', choices=LOSS_NAMES)
    parser.add_argument('--alpha', default=0.5, type=float, help='flow loss weight')
    parser.add_argument('--flow_loss', default='smoothl1', choices=['smoothl1', 'l1', 'mse'])
    parser.add_argument('--decomp_weight', default=0.05, type=float)
    parser.add_argument('--ctx_weight', default=0.05, type=float)
    parser.add_argument('--ctx_smooth_weight', default=0.01, type=float)

    # dataset
    parser.add_argument('--dataset', default='ISIC', help='dataset folder name under inputs/ or absolute dataset path')
    parser.add_argument('--img_ext', default='.png')
    parser.add_argument('--mask_ext', default='.png')
    parser.add_argument('--split_dir', default=None, help='folder containing train.txt and val.txt')
    parser.add_argument('--val_ratio', default=0.2, type=float)

    # optimizer
    parser.add_argument('--optimizer', default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--milestones', default='40,70', type=str)
    parser.add_argument('--gamma_scheduler', default=0.5, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N')

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--cfg', type=str, default=None)

    return parser.parse_args()



def extract_main_output(output):
    if isinstance(output, dict):
        return output['logits']
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output



def build_criterion(config, device):
    if config['loss'] == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss().to(device)
    if config['loss'] in ['C2UNetLoss', 'C2SegLoss']:
        return losses.__dict__[config['loss']](
            alpha=config['alpha'],
            flow_loss=config['flow_loss'],
            decomp_weight=config['decomp_weight'],
            ctx_weight=config['ctx_weight'],
            ctx_smooth_weight=config['ctx_smooth_weight'],
        ).to(device)
    if config['loss'] == 'BoundaryBCEDiceLoss':
        return losses.BoundaryBCEDiceLoss(
            boundary_weight=1.0,
            sigma=0.1,
        ).to(device)
    return losses.__dict__[config['loss']]().to(device)



def build_model(config):
    if config['arch'] not in archs.__dict__:
        raise ValueError(f"Unknown architecture: {config['arch']}")
    return archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        num_stages=config['num_stages'],
        feat_dim=config['feat_dim'],
        ctx_dim=config['ctx_dim'],
        hidden_dim=config['hidden_dim'],
        tau=config['tau'],
        beta=config.get('beta', None),
        eps=config['eps'],
        lambda1=config['lambda1'],
        lambda2=config['lambda2'],
        lambda3=config['lambda3'],
        sigma=config['sigma'],
        eta_init=config['eta_init'],
        gamma_init=config['gamma_init'],
        bacr_prox=config.get('bacr_prox', 'cnn'),
        dcnv4_group=config.get('dcnv4_group', 4),
        dcnv4_offset_scale=config.get('dcnv4_offset_scale', 0.5),
        dcnv4_dw_kernel_size=config.get('dcnv4_dw_kernel_size', 3),
        dcnv4_center_feature_scale=config.get('dcnv4_center_feature_scale', True),
        bacr_boundary_gate=config.get('bacr_boundary_gate', True),
        bacr_dcn_start_stage=config.get('bacr_dcn_start_stage', 1),
        bacr_dcn_res_scale_init=config.get('bacr_dcn_res_scale_init', 0.05),
        pretrained=config.get('pretrained', False),
        pretrained_ckpt=config.get('pretrained_ckpt', None),
    )



def build_optimizer(config, model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        return optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    if config['optimizer'] == 'AdamW':
        return optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    return optim.SGD(
        params,
        lr=config['lr'],
        momentum=config['momentum'],
        nesterov=config['nesterov'],
        weight_decay=config['weight_decay'],
    )



def build_scheduler(config, optimizer):
    if config['scheduler'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    if config['scheduler'] == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config['factor'],
            patience=config['patience'],
            verbose=1,
            min_lr=config['min_lr'],
        )
    if config['scheduler'] == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config['milestones'].split(',')],
            gamma=config['gamma_scheduler'],
        )
    return None



def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def compute_loss(criterion, output, target):
    structured = isinstance(criterion, (losses.C2UNetLoss, losses.C2SegLoss))
    loss_input = output if structured else extract_main_output(output)
    loss_out = criterion(loss_input, target)
    if isinstance(loss_out, dict):
        return loss_out['loss'], loss_out
    return loss_out, {
        'loss': loss_out.detach(),
        'seg': torch.tensor(0.0, device=loss_out.device),
        'flow': torch.tensor(0.0, device=loss_out.device),
        'decomp': torch.tensor(0.0, device=loss_out.device),
        'ctx': torch.tensor(0.0, device=loss_out.device),
        'ctx_smooth': torch.tensor(0.0, device=loss_out.device),
    }



def train_one_epoch(config, train_loader, model, criterion, optimizer, device):
    meters = {
        'loss': AverageMeter(),
        'seg': AverageMeter(),
        'flow': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'decomp': AverageMeter(),
        'ctx': AverageMeter(),
        'ctx_smooth': AverageMeter(),
    }

    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)
        loss, loss_dict = compute_loss(criterion, output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = compute_seg_metrics(extract_main_output(output), target)
        bs = input.size(0)
        meters['loss'].update(float(loss.item()), bs)
        meters['seg'].update(float(loss_dict['seg'].item()), bs)
        meters['flow'].update(float(loss_dict['flow'].item()), bs)
        meters['iou'].update(metrics['iou'], bs)
        meters['dice'].update(metrics['dice'], bs)
        meters['decomp'].update(float(loss_dict.get('decomp', torch.tensor(0.0, device=loss.device)).item()), bs)
        meters['ctx'].update(float(loss_dict.get('ctx', torch.tensor(0.0, device=loss.device)).item()), bs)
        meters['ctx_smooth'].update(float(loss_dict.get('ctx_smooth', torch.tensor(0.0, device=loss.device)).item()), bs)

        postfix = OrderedDict([
            ('loss', meters['loss'].avg),
            ('seg', meters['seg'].avg),
            ('flow', meters['flow'].avg),
            ('iou', meters['iou'].avg),
            ('dice', meters['dice'].avg),
            ('dec', meters['decomp'].avg),
            ('ctx', meters['ctx'].avg),
            ('ctx_s', meters['ctx_smooth'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([(k, v.avg) for k, v in meters.items()])



def validate(config, val_loader, model, criterion, device):
    meters = {
        'loss': AverageMeter(),
        'seg': AverageMeter(),
        'flow': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'hd95': AverageMeter(),
        'assd': AverageMeter(),
        'bd': AverageMeter(),
        'bdsd': AverageMeter(),
        's_alpha': AverageMeter(),
        'decomp': AverageMeter(),
        'ctx': AverageMeter(),
        'ctx_smooth': AverageMeter(),
    }

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            loss, loss_dict = compute_loss(criterion, output, target)
            metrics = compute_seg_metrics(extract_main_output(output), target)
            bs = input.size(0)

            meters['loss'].update(float(loss.item()), bs)
            meters['seg'].update(float(loss_dict['seg'].item()), bs)
            meters['flow'].update(float(loss_dict['flow'].item()), bs)
            meters['iou'].update(metrics['iou'], bs)
            meters['dice'].update(metrics['dice'], bs)
            meters['hd95'].update(metrics['hd95'], bs)
            meters['assd'].update(metrics['assd'], bs)
            meters['bd'].update(metrics['bd'], bs)
            meters['bdsd'].update(metrics['bdsd'], bs)
            meters['s_alpha'].update(metrics['s_alpha'], bs)
            meters['decomp'].update(float(loss_dict.get('decomp', torch.tensor(0.0, device=loss.device)).item()), bs)
            meters['ctx'].update(float(loss_dict.get('ctx', torch.tensor(0.0, device=loss.device)).item()), bs)
            meters['ctx_smooth'].update(float(loss_dict.get('ctx_smooth', torch.tensor(0.0, device=loss.device)).item()), bs)

            postfix = OrderedDict([
                ('loss', meters['loss'].avg),
                ('iou', meters['iou'].avg),
                ('dice', meters['dice'].avg),
                ('hd95', meters['hd95'].avg),
                ('assd', meters['assd'].avg),
                ('bd', meters['bd'].avg),
                ('bdsd', meters['bdsd'].avg),
                ('S_a', meters['s_alpha'].avg),
                ('dec', meters['decomp'].avg),
                ('ctx', meters['ctx'].avg),
                ('ctx_s', meters['ctx_smooth'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([(k, v.avg) for k, v in meters.items()])



def main():
    config = vars(parse_args())
    seed_torch(config['seed'])

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'

    dataset_dir = resolve_dataset_dir(config['dataset'], base_dir)
    img_dir = dataset_dir / 'images'
    mask_dir = dataset_dir / 'masks'

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f'Dataset must contain images/ and masks/ folders: {dataset_dir}')

    config['dataset'] = str(dataset_dir)
    config['img_ext'] = find_existing_ext(img_dir, config['img_ext'])
    config['mask_ext'] = find_existing_ext(mask_dir, config['mask_ext'])

    dataset_name = dataset_dir.name
    if config['name'] is None:
        config['name'] = f'{dataset_name}_{config["arch"]}'
    config['name'] = Path(str(config['name'])).name

    save_dir = models_dir / config['name']
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'config.yml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(config).to(device)
    trainable_params = count_params(model) / 1e6
    print(f'Trainable Params: {trainable_params:.2f} M')
    print(f'Dataset dir: {dataset_dir}')
    print(f'Image ext: {config["img_ext"]} | Mask ext: {config["mask_ext"]}')
    print(f'Split source: {describe_split_source(base_dir, dataset_dir, config.get("split_dir"))}')

    criterion = build_criterion(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    img_ids = list_image_ids(img_dir, config['img_ext'])
    if len(img_ids) == 0:
        raise RuntimeError(f'No images found under {img_dir} with extension {config["img_ext"]}')
    train_img_ids, val_img_ids = build_splits(
        base_dir=base_dir,
        dataset_dir=dataset_dir,
        img_ids=img_ids,
        split_dir=config.get('split_dir'),
        val_ratio=config['val_ratio'],
        seed=41,
    )
    print(f'Train images: {len(train_img_ids)} | Val images: {len(val_img_ids)}')

    train_transform = Compose([
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=str(img_dir),
        mask_dir=str(mask_dir),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform,
    )
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=str(img_dir),
        mask_dir=str(mask_dir),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(1, min(config['batch_size'], 4)),
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True,
    )

    log = OrderedDict([
        ('epoch', []), ('lr', []),
        ('loss', []), ('seg', []), ('flow', []), ('decomp', []), ('ctx', []), ('ctx_smooth', []), ('iou', []), ('dice', []),
        ('val_loss', []), ('val_seg', []), ('val_flow', []), ('val_decomp', []), ('val_ctx', []), ('val_ctx_smooth', []),
        ('val_iou', []), ('val_dice', []), ('val_hd95', []), ('val_assd', []), ('val_bd', []), ('val_bdsd', []), ('val_s_alpha', []),
    ])

    best_iou = 0.0
    trigger = 0
    for epoch in range(config['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{config["epochs"]}] lr={current_lr:.8f}')

        train_log = train_one_epoch(config, train_loader, model, criterion, optimizer, device)
        val_log = validate(config, val_loader, model, criterion, device)

        if scheduler is not None:
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])
            else:
                scheduler.step()

        print(
            'train loss %.4f - iou %.4f - dice %.4f | '
            'val loss %.4f - val_iou %.4f - val_dice %.4f - val_hd95 %.4f - val_assd %.4f - val_bd %.4f - val_bdsd %.4f - val_Salpha %.4f'
            % (
                train_log['loss'], train_log['iou'], train_log['dice'],
                val_log['loss'], val_log['iou'], val_log['dice'], val_log['hd95'], val_log['assd'], val_log['bd'], val_log['bdsd'], val_log['s_alpha']
            )
        )

        log['epoch'].append(epoch + 1)
        log['lr'].append(current_lr)
        for k in ['loss', 'seg', 'flow', 'decomp', 'ctx', 'ctx_smooth', 'iou', 'dice']:
            log[k].append(train_log[k])
        for k in ['loss', 'seg', 'flow', 'decomp', 'ctx', 'ctx_smooth', 'iou', 'dice', 'hd95', 'assd', 'bd', 'bdsd', 's_alpha']:
            log[f'val_{k}'].append(val_log[k])

        pd.DataFrame(log).to_csv(save_dir / 'log.csv', index=False)

        trigger += 1
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), save_dir / 'model.pth')
            best_iou = val_log['iou']
            print('=> saved best model')
            trigger = 0

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print('=> early stopping')
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

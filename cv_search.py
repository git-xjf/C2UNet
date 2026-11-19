import argparse
import itertools
import json
import os
import random
import shlex
import subprocess
from collections import OrderedDict
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations import Compose, HorizontalFlip, Normalize, RandomRotate90, Resize, VerticalFlip
from sklearn.model_selection import KFold

from data_utils import find_existing_ext, list_image_ids, resolve_dataset_dir
from dataset import Dataset
from train import (
    ARCH_NAMES,
    LOSS_NAMES,
    build_criterion,
    build_model,
    build_optimizer,
    build_scheduler,
    seed_torch,
    train_one_epoch,
    validate,
)
from utils import str2bool


DEFAULT_SEARCH_SPACE = OrderedDict([
    # Keep the default space intentionally compact and centered around the user's current run.
    # Use --search_space search_space.json to replace/extend this for larger experiments.
    ('lr', [1e-4, 7e-5, 5e-5]),
    ('lambda1', [0.5, 0.6, 0.7]),
    ('lambda2', [0.20, 0.25, 0.30]),
    ('beta', [0.4, 0.6, 0.8]),
    ('alpha', [0.5, 0.6, 0.7]),
    ('decomp_weight', [0.03, 0.05, 0.07]),
    ('ctx_weight', [0.02, 0.03, 0.04]),
    ('ctx_smooth_weight', [0.001, 0.002, 0.004]),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description='K-fold random/grid hyper-parameter search for C2UNet training.'
    )

    # Search-specific arguments.
    parser.add_argument('--cv_folds', default=3, type=int, help='number of CV folds')
    parser.add_argument('--cv_epochs', default=80, type=int, help='epochs for each CV fold/trial')
    parser.add_argument('--num_trials', default=12, type=int, help='number of sampled trials for random search')
    parser.add_argument('--search_mode', default='random', choices=['random', 'grid'])
    parser.add_argument('--search_space', default=None, type=str,
                        help='optional JSON/YAML file, e.g. {"lr": [0.0001, 0.00005], "alpha": [0.5, 0.6]}')
    parser.add_argument('--cv_name', default=None, type=str, help='output folder under models/')
    parser.add_argument('--metric', default='iou', choices=['iou', 'dice'], help='metric used to select the best trial')
    parser.add_argument('--save_fold_models', default=False, type=str2bool,
                        help='save best model.pth for every trial/fold; usually False to save disk')
    parser.add_argument('--final_epochs', default=None, type=int,
                        help='epochs written into run_final_best.sh; default uses --epochs')
    parser.add_argument('--final_name', default=None, type=str,
                        help='model name for the final full training command')
    parser.add_argument('--run_final', default=False, type=str2bool,
                        help='after CV, immediately launch train.py with best hyper-parameters')

    # Base training arguments copied from train.py so the original training command can be reused.
    parser.add_argument('--name', default=None, help='base model name')
    parser.add_argument('--epochs', default=400, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N')

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
    parser.add_argument('--dcnv4_group', default=4, type=int)
    parser.add_argument('--dcnv4_offset_scale', default=0.5, type=float)
    parser.add_argument('--dcnv4_dw_kernel_size', default=3, type=int)
    parser.add_argument('--dcnv4_center_feature_scale', default=True, type=str2bool)
    parser.add_argument('--bacr_boundary_gate', default=True, type=str2bool)
    parser.add_argument('--bacr_dcn_start_stage', default=1, type=int)
    parser.add_argument('--bacr_dcn_res_scale_init', default=0.05, type=float)
    parser.add_argument('--pretrained', default=True, type=str2bool)
    parser.add_argument('--pretrained_ckpt', default=None, type=str)

    parser.add_argument('--loss', default='C2UNetLoss', choices=LOSS_NAMES)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--flow_loss', default='smoothl1', choices=['smoothl1', 'l1', 'mse'])
    parser.add_argument('--decomp_weight', default=0.05, type=float)
    parser.add_argument('--ctx_weight', default=0.05, type=float)
    parser.add_argument('--ctx_smooth_weight', default=0.01, type=float)

    parser.add_argument('--dataset', default='ISIC')
    parser.add_argument('--img_ext', default='.png')
    parser.add_argument('--mask_ext', default='.png')
    parser.add_argument('--split_dir', default=None)
    parser.add_argument('--val_ratio', default=0.2, type=float)

    parser.add_argument('--optimizer', default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--milestones', default='40,70', type=str)
    parser.add_argument('--gamma_scheduler', default=0.5, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N')

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=3407, type=int)

    return parser.parse_args()


def load_search_space(path):
    if path is None:
        return DEFAULT_SEARCH_SPACE.copy()
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix.lower() in {'.yml', '.yaml'}:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError('--search_space must contain a non-empty dict of parameter -> list of values.')
    space = OrderedDict()
    for key, values in data.items():
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f'Search space for {key!r} must be a non-empty list.')
        space[key] = list(values)
    return space


def make_trials(space, mode, num_trials, seed):
    keys = list(space.keys())
    all_combos = [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]
    if mode == 'grid':
        return all_combos[:num_trials] if num_trials > 0 else all_combos
    rng = random.Random(seed)
    if num_trials >= len(all_combos):
        rng.shuffle(all_combos)
        return all_combos
    return rng.sample(all_combos, num_trials)


def build_transforms(config):
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
    return train_transform, val_transform


def make_loader(config, img_ids, img_dir, mask_dir, train, transform):
    dataset = Dataset(
        img_ids=img_ids,
        img_dir=str(img_dir),
        mask_dir=str(mask_dir),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'] if train else max(1, min(config['batch_size'], 4)),
        shuffle=train,
        num_workers=config['num_workers'],
        drop_last=train,
        pin_memory=True,
    )


def run_one_fold(config, train_ids, val_ids, img_dir, mask_dir, fold_dir, save_model=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    train_transform, val_transform = build_transforms(config)
    train_loader = make_loader(config, train_ids, img_dir, mask_dir, True, train_transform)
    val_loader = make_loader(config, val_ids, img_dir, mask_dir, False, val_transform)

    model = build_model(config).to(device)
    criterion = build_criterion(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    fold_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    best_row = None
    rows = []
    trigger = 0

    for epoch in range(config['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Trial {config['trial_id']} Fold {config['fold_id']} | Epoch [{epoch + 1}/{config['epochs']}] lr={current_lr:.8f}")
        train_log = train_one_epoch(config, train_loader, model, criterion, optimizer, device)
        val_log = validate(config, val_loader, model, criterion, device)

        if scheduler is not None:
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])
            else:
                scheduler.step()

        row = OrderedDict([
            ('trial', config['trial_id']),
            ('fold', config['fold_id']),
            ('epoch', epoch + 1),
            ('lr', current_lr),
            ('loss', train_log['loss']),
            ('iou', train_log['iou']),
            ('dice', train_log['dice']),
            ('val_loss', val_log['loss']),
            ('val_iou', val_log['iou']),
            ('val_dice', val_log['dice']),
            ('val_hd95', val_log['hd95']),
            ('val_assd', val_log['assd']),
            ('val_bd', val_log['bd']),
            ('val_bdsd', val_log['bdsd']),
            ('val_s_alpha', val_log['s_alpha']),
        ])
        rows.append(row)
        pd.DataFrame(rows).to_csv(fold_dir / 'log.csv', index=False)

        metric_key = f"val_{config['metric']}"
        current_metric = row[metric_key]
        trigger += 1
        if current_metric > best_metric:
            best_metric = current_metric
            best_row = dict(row)
            if save_model:
                torch.save(model.state_dict(), fold_dir / 'model.pth')
            trigger = 0
            print(f'=> fold best {config["metric"]}: {best_metric:.5f}')

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print('=> early stopping')
            break

        torch.cuda.empty_cache()

    del model, optimizer, scheduler, criterion, train_loader, val_loader
    torch.cuda.empty_cache()
    return best_row


def shell_value(value):
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if value is None:
        return None
    return str(value)


def build_final_command(config):
    ordered_args = [
        'dataset', 'arch', 'name', 'loss', 'batch_size', 'epochs',
        'input_h', 'input_w', 'num_stages', 'feat_dim', 'ctx_dim', 'hidden_dim',
        'tau', 'beta', 'eps', 'lambda1', 'lambda2', 'lambda3', 'sigma', 'eta_init', 'gamma_init',
        'bacr_prox', 'dcnv4_group', 'dcnv4_offset_scale', 'dcnv4_dw_kernel_size',
        'dcnv4_center_feature_scale', 'bacr_boundary_gate', 'bacr_dcn_start_stage', 'bacr_dcn_res_scale_init',
        'pretrained', 'pretrained_ckpt',
        'alpha', 'flow_loss', 'decomp_weight', 'ctx_weight', 'ctx_smooth_weight',
        'optimizer', 'lr', 'weight_decay', 'scheduler', 'min_lr',
        'img_ext', 'mask_ext', 'num_workers', 'seed',
    ]
    parts = ['python train.py']
    for key in ordered_args:
        value = shell_value(config.get(key))
        if value is None:
            continue
        parts.append(f"  --{key} {shlex.quote(value)}")
    return ' \\\n'.join(parts)


def main():
    args = parse_args()
    base_config = vars(args)
    seed_torch(base_config['seed'])

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'

    dataset_dir = resolve_dataset_dir(base_config['dataset'], base_dir)
    img_dir = dataset_dir / 'images'
    mask_dir = dataset_dir / 'masks'
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f'Dataset must contain images/ and masks/ folders: {dataset_dir}')

    base_config['dataset'] = str(dataset_dir)
    base_config['img_ext'] = find_existing_ext(img_dir, base_config['img_ext'])
    base_config['mask_ext'] = find_existing_ext(mask_dir, base_config['mask_ext'])

    dataset_name = dataset_dir.name
    base_name = Path(str(base_config['name'] or f'{dataset_name}_{base_config["arch"]}')).name
    cv_name = Path(str(base_config['cv_name'] or f'{base_name}_cv_search')).name
    cv_dir = models_dir / cv_name
    cv_dir.mkdir(parents=True, exist_ok=True)

    img_ids = list_image_ids(img_dir, base_config['img_ext'])
    if len(img_ids) == 0:
        raise RuntimeError(f'No images found under {img_dir} with extension {base_config["img_ext"]}')
    if base_config['cv_folds'] < 2:
        raise ValueError('--cv_folds must be >= 2')
    if len(img_ids) < base_config['cv_folds']:
        raise ValueError(f'Not enough images ({len(img_ids)}) for {base_config["cv_folds"]} folds')

    space = load_search_space(base_config['search_space'])
    trials = make_trials(space, base_config['search_mode'], base_config['num_trials'], base_config['seed'])

    print(f'Dataset dir: {dataset_dir}')
    print(f'Images: {len(img_ids)} | folds: {base_config["cv_folds"]} | trials: {len(trials)} | cv_epochs: {base_config["cv_epochs"]}')
    print(f'Output dir: {cv_dir}')
    print('Search space:')
    print(yaml.safe_dump(dict(space), sort_keys=False, allow_unicode=True))

    with open(cv_dir / 'base_config.yml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(base_config, f, sort_keys=False, allow_unicode=True)
    with open(cv_dir / 'search_space.yml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(dict(space), f, sort_keys=False, allow_unicode=True)

    kfold = KFold(n_splits=base_config['cv_folds'], shuffle=True, random_state=base_config['seed'])
    all_results = []

    for trial_idx, hp in enumerate(trials, start=1):
        trial_id = f'{trial_idx:03d}'
        print('\n' + '=' * 80)
        print(f'Trial {trial_id}/{len(trials)} hyper-parameters: {hp}')
        trial_fold_rows = []

        for fold_idx, (train_index, val_index) in enumerate(kfold.split(img_ids), start=1):
            fold_config = base_config.copy()
            fold_config.update(hp)
            fold_config['epochs'] = base_config['cv_epochs']
            fold_config['metric'] = base_config['metric']
            fold_config['trial_id'] = trial_id
            fold_config['fold_id'] = fold_idx
            fold_config['name'] = f'{cv_name}/trial_{trial_id}_fold_{fold_idx}'
            fold_config['seed'] = int(base_config['seed']) + trial_idx * 100 + fold_idx
            seed_torch(fold_config['seed'])

            train_ids = [img_ids[i] for i in train_index]
            val_ids = [img_ids[i] for i in val_index]
            fold_dir = cv_dir / f'trial_{trial_id}' / f'fold_{fold_idx}'

            print(f'Fold {fold_idx}/{base_config["cv_folds"]}: train={len(train_ids)} val={len(val_ids)}')
            best_row = run_one_fold(
                fold_config,
                train_ids=train_ids,
                val_ids=val_ids,
                img_dir=img_dir,
                mask_dir=mask_dir,
                fold_dir=fold_dir,
                save_model=base_config['save_fold_models'],
            )
            for key, value in hp.items():
                best_row[key] = value
            trial_fold_rows.append(best_row)

        fold_df = pd.DataFrame(trial_fold_rows)
        summary = OrderedDict()
        summary['trial'] = trial_id
        for key, value in hp.items():
            summary[key] = value
        for metric_name in ['val_iou', 'val_dice', 'val_loss', 'val_hd95', 'val_assd', 'val_bd', 'val_bdsd', 'val_s_alpha']:
            summary[f'mean_{metric_name}'] = float(fold_df[metric_name].mean())
            summary[f'std_{metric_name}'] = float(fold_df[metric_name].std(ddof=0))
        all_results.append(summary)

        result_df = pd.DataFrame(all_results)
        sort_col = f'mean_val_{base_config["metric"]}'
        result_df = result_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        result_df.to_csv(cv_dir / 'cv_results.csv', index=False)
        fold_df.to_csv(cv_dir / f'trial_{trial_id}_fold_best.csv', index=False)
        print(f'Trial {trial_id} mean val_{base_config["metric"]}: {summary[sort_col]:.5f}')
        print(f'Current best trial: {result_df.iloc[0]["trial"]} | {sort_col}={result_df.iloc[0][sort_col]:.5f}')

    results = pd.DataFrame(all_results)
    sort_col = f'mean_val_{base_config["metric"]}'
    tie_col = 'mean_val_dice' if base_config['metric'] == 'iou' else 'mean_val_iou'
    results = results.sort_values([sort_col, tie_col], ascending=False).reset_index(drop=True)
    results.to_csv(cv_dir / 'cv_results.csv', index=False)

    best = results.iloc[0].to_dict()
    best_hp = {key: best[key] for key in space.keys()}
    final_config = base_config.copy()
    final_config.update(best_hp)
    final_config['epochs'] = base_config['final_epochs'] or base_config['epochs']
    final_config['name'] = base_config['final_name'] or f'{base_name}_bestcv'

    with open(cv_dir / 'best_config.yml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(final_config, f, sort_keys=False, allow_unicode=True)
    with open(cv_dir / 'best_summary.yml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(best, f, sort_keys=False, allow_unicode=True)

    command = build_final_command(final_config)
    run_script = cv_dir / 'run_final_best.sh'
    with open(run_script, 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env bash\nset -e\n\n')
        f.write('SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\n')
        f.write('PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)\n')
        f.write('cd "$PROJECT_DIR"\n\n')
        f.write(command)
        f.write('\n')
    os.chmod(run_script, 0o755)

    print('\n' + '=' * 80)
    print('Best hyper-parameters:')
    print(yaml.safe_dump(best_hp, sort_keys=False, allow_unicode=True))
    print(f'Best CV {sort_col}: {best[sort_col]:.5f}')
    print(f'Saved: {cv_dir / "cv_results.csv"}')
    print(f'Saved: {cv_dir / "best_config.yml"}')
    print(f'Saved final command: {run_script}')
    print('\nFinal training command:')
    print(command)

    if base_config['run_final']:
        print('\n=> launching final training with best hyper-parameters')
        subprocess.run(['bash', str(run_script)], cwd=str(base_dir), check=True)


if __name__ == '__main__':
    main()

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations import Compose, Normalize, Resize
from tqdm import tqdm

import archs
from data_utils import build_splits, describe_split_source, find_existing_ext, list_image_ids, resolve_dataset_dir
from dataset import Dataset
from metrics import compute_seg_metrics
from utils import AverageMeter, count_params

try:
    from thop import profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False


class FinalLogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out['logits']
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='model name, e.g. Kvasir-SEG_C2UNetResNet50')
    parser.add_argument('--split_dir', default=None, help='optional folder containing train.txt and val.txt')
    return parser.parse_args()



def extract_main_output(output):
    if isinstance(output, dict):
        return output['logits']
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output



def build_model(config):
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config.get('deep_supervision', True),
        num_stages=config.get('num_stages', 3),
        feat_dim=config.get('feat_dim', 256),
        ctx_dim=config.get('ctx_dim', 128),
        hidden_dim=config.get('hidden_dim', 128),
        tau=config.get('tau', 0.5),
        beta=config.get('beta', None),
        eps=config.get('eps', 1.0),
        lambda1=config.get('lambda1', 0.5),
        lambda2=config.get('lambda2', 0.25),
        lambda3=config.get('lambda3', 1.0),
        sigma=config.get('sigma', 0.15),
        eta_init=config.get('eta_init', 1.0),
        gamma_init=config.get('gamma_init', 1.0),
        bacr_prox=config.get('bacr_prox', 'cnn'),
        dcnv4_group=config.get('dcnv4_group', 4),
        dcnv4_offset_scale=config.get('dcnv4_offset_scale', 0.5),
        dcnv4_dw_kernel_size=config.get('dcnv4_dw_kernel_size', 3),
        dcnv4_center_feature_scale=config.get('dcnv4_center_feature_scale', True),
        bacr_boundary_gate=config.get('bacr_boundary_gate', True),
        bacr_dcn_start_stage=config.get('bacr_dcn_start_stage', 1),
        bacr_dcn_res_scale_init=config.get('bacr_dcn_res_scale_init', 0.05),
    )
    return model



def compute_flops(model, input_shape, device):
    if not THOP_AVAILABLE:
        return None
    wrapper = FinalLogitsWrapper(model).to(device)
    wrapper.eval()
    dummy = torch.randn(*input_shape, device=device)
    flops, _ = profile(wrapper, inputs=(dummy,), verbose=False)
    return flops / 1e9



def benchmark_inference(model, input_shape, device, warmup=20, iters=100):
    wrapper = FinalLogitsWrapper(model).to(device)
    wrapper.eval()
    x = torch.randn(*input_shape, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = wrapper(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(iters):
            _ = wrapper(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

    total = end - start
    latency = total / iters
    fps = 1.0 / max(latency, 1e-12)
    return latency, fps



def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'
    outputs_dir = base_dir / 'outputs'

    model_name = Path(str(args.name)).name
    model_dir = models_dir / model_name
    output_dir = outputs_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / 'config.yml'
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['name'] = model_name

    dataset_dir = resolve_dataset_dir(str(config['dataset']), base_dir)
    img_dir = dataset_dir / 'images'
    mask_dir = dataset_dir / 'masks'
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f'Dataset must contain images/ and masks/ folders: {dataset_dir}')

    config['img_ext'] = find_existing_ext(img_dir, config.get('img_ext', '.png'))
    config['mask_ext'] = find_existing_ext(mask_dir, config.get('mask_ext', '.png'))

    print('-' * 20)
    for key in config.keys():
        print(f'{key}: {config[key]}')
    print('-' * 20)
    print(f'Model dir: {model_dir}')
    print(f'Dataset dir: {dataset_dir}')
    print(f'Image ext: {config["img_ext"]} | Mask ext: {config["mask_ext"]}')
    print(f'Split source: {describe_split_source(base_dir, dataset_dir, args.split_dir or config.get("split_dir"))}')
    print(f'Saving predictions to: {output_dir}')

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'=> creating model {config["arch"]}')
    model = build_model(config).to(device)

    ckpt_path = model_dir / 'model.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    img_ids = list_image_ids(img_dir, config['img_ext'])
    _, val_img_ids = build_splits(
        base_dir=base_dir,
        dataset_dir=dataset_dir,
        img_ids=img_ids,
        split_dir=args.split_dir or config.get('split_dir'),
        val_ratio=config.get('val_ratio', 0.2),
        seed=41,
    )
    print(f'Validation images: {len(val_img_ids)}')

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=str(img_dir),
        mask_dir=str(mask_dir),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(1, min(config['batch_size'], 4)),
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True,
    )

    meters = {
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'hd95': AverageMeter(),
        'assd': AverageMeter(),
        'bd': AverageMeter(),
        'bdsd': AverageMeter(),
        's_alpha': AverageMeter(),
    }

    for c in range(config['num_classes']):
        (output_dir / str(c)).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            logits = extract_main_output(output)
            metrics = compute_seg_metrics(logits, target)

            for key in meters:
                meters[key].update(metrics[key], input.size(0))

            prob = torch.sigmoid(logits).cpu().numpy()
            prob[prob >= 0.5] = 1
            prob[prob < 0.5] = 0
            for i in range(len(prob)):
                for c in range(config['num_classes']):
                    save_path = output_dir / str(c) / f'{meta["img_id"][i]}.png'
                    cv2.imwrite(str(save_path), (prob[i, c] * 255).astype('uint8'))

    params_m = count_params(model) / 1e6
    flops_g = compute_flops(model, (1, config['input_channels'], config['input_h'], config['input_w']), device)
    latency, fps = benchmark_inference(
        model,
        (1, config['input_channels'], config['input_h'], config['input_w']),
        device,
        warmup=20,
        iters=100,
    )

    print(f'Params (M): {params_m:.2f}')
    if flops_g is not None:
        print(f'FLOPs (G): {flops_g:.2f}')
    else:
        print('FLOPs (G): unavailable (install thop to enable)')
    print(f'Single-image inference time (s): {latency:.6f}')
    print(f'FPS: {fps:.2f}')
    print(f'IoU: {meters["iou"].avg:.4f}')
    print(f'Dice: {meters["dice"].avg:.4f}')
    print(f'HD95: {meters["hd95"].avg:.4f}')
    print(f'ASSD: {meters["assd"].avg:.4f}')
    print(f'BD: {meters["bd"].avg:.4f}')
    print(f'BDSD: {meters["bdsd"].avg:.4f}')
    print(f'S_alpha: {meters["s_alpha"].avg:.4f}')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

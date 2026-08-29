"""Model-only latency, parameter, and MAC measurement with raw trace output."""

import argparse
import json
import platform
import time

import numpy as np
import torch
from torch.cuda.amp import autocast

from config_loader import load_config
from models.student import LiteMedFormerS
from utils import load_checkpoint, sha256_file


def main():
    parser = argparse.ArgumentParser(description="Benchmark exported student inference")
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--output', default='latency_trace.json')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = LiteMedFormerS(
        in_channels=config['data']['in_channels'],
        num_classes=config['data']['num_classes'],
        encoder_channels=config['model']['student']['encoder_channels'],
        decoder_channels=config['model']['student']['decoder_channels'],
    ).to(device).eval()
    load_checkpoint(args.checkpoint, model, strict=True)

    height, width = config['data']['image_size']
    sample = torch.randn(
        1, config['data']['in_channels'], height, width, device=device
    )
    amp_enabled = bool(config['training']['mixed_precision'] and device.type == 'cuda')

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            with autocast(enabled=amp_enabled):
                model(sample)
        synchronize()

        trace_ms = []
        for _ in range(args.iterations):
            synchronize()
            start = time.perf_counter_ns()
            with autocast(enabled=amp_enabled):
                model(sample)
            synchronize()
            trace_ms.append((time.perf_counter_ns() - start) / 1e6)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    mac_count = None
    try:
        from thop import profile
        mac_count, _ = profile(model, inputs=(sample,), verbose=False)
        mac_count = int(mac_count)
    except ImportError:
        pass

    trace = np.asarray(trace_ms, dtype=np.float64)
    output = {
        'scope': 'model_only_forward_pass',
        'input_shape': list(sample.shape),
        'warmup_iterations': args.warmup,
        'measured_iterations': args.iterations,
        'mixed_precision': amp_enabled,
        'device': str(device),
        'device_name': (
            torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor()
        ),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'platform': platform.platform(),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'parameters': int(parameter_count),
        'macs': mac_count,
        'median_ms': float(np.median(trace)),
        'q1_ms': float(np.quantile(trace, 0.25)),
        'q3_ms': float(np.quantile(trace, 0.75)),
        'raw_trace_ms': trace_ms,
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({key: value for key, value in output.items() if key != 'raw_trace_ms'}, indent=2))


if __name__ == '__main__':
    main()

"""
Cross-platform utilities for BrushLegacy
"""
import platform
import torch
import numpy as np
import yaml
from pathlib import Path

def detect_device():
    """Auto-detect best available device"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")  # MacBook M1/M2
    elif torch.cuda.is_available():
        return torch.device("cuda")  # RT box with GPU
    else:
        return torch.device("cpu")   # fallback

def load_config(config_path="config.yaml"):
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return get_default_config()

def get_default_config():
    """Default configuration if file is missing"""
    return {
        'models': {
            'painttransformer': 'models/220_net_g.pth',
            'u2net': 'models/u2net.pth'
        },
        'canvas': {
            'width_px': 512,
            'height_px': 384,
            'width_mm': 400,
            'height_mm': 300
        },
        'stroke': {
            'patch_size': 32,
            'stroke_num': 8,
            'layers': [3, 4, 5],
            'p_dim': 16
        },
        'device': 'auto'
    }

def ensure_dirs():
    """Create necessary directories"""
    dirs = ['data/input', 'data/strokes', 'data/trajectories', 'data/output', 'models']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_strokes(strokes, output_path):
    """Save stroke data in NPZ format"""
    np.savez_compressed(output_path, **strokes)
    print(f"Strokes saved to: {output_path}")

def load_strokes(input_path):
    """Load stroke data from NPZ format"""
    data = np.load(input_path, allow_pickle=True)
    return {key: data[key] for key in data.files}

def get_platform_info():
    """Get platform information for debugging"""
    info = {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'device': str(detect_device())
    }
    return info

def print_system_info():
    """Print system information"""
    info = get_platform_info()
    print("=" * 50)
    print("BrushLegacy System Information")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key:10}: {value}")
    print("=" * 50)
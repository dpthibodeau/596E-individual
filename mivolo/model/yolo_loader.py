import torch
from ultralytics import YOLO
from torch import serialization
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel

def load_yolo_model(weights_path):
    """Custom loader for YOLO model that handles PyTorch 2.6+ security changes"""
    
    # Add necessary safe globals
    serialization.add_safe_globals([
        DetectionModel,
        Conv,
        torch.nn.modules.container.Sequential,
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
        torch.nn.SiLU,
        torch.nn.MaxPool2d,
        torch.nn.Module,
        torch.nn.ModuleList
    ])
    
    # Override torch load temporarily
    original_load = torch.load
    def custom_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = custom_load
    
    try:
        model = YOLO(weights_path)
    finally:
        # Restore original torch.load
        torch.load = original_load
    
    return model
import torch
from torch import serialization
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
import os
import numpy as np
import cv2
from .structures import PersonAndFaceResult

class Detector:
    def __init__(self, weights, device="cuda:0", verbose=True):
        if verbose:
            print(f"Loading YOLO model from {weights}")
        self.device = device
        
        # Verify weights file exists
        if not os.path.exists(weights):
            raise FileNotFoundError(f"YOLO weights file not found: {weights}")
        
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
        
        # Store original torch load function
        _torch_load_original = torch.load
        
        def safe_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return _torch_load_original(*args, weights_only=False, **kwargs)
        
        # Replace torch.load temporarily
        torch.load = safe_load
        
        try:
            self.yolo = YOLO(weights)
            self.yolo.to(device)
        finally:
            # Restore original torch.load
            torch.load = _torch_load_original
            
        if verbose:
            print("YOLO model loaded successfully")

    def predict(self, images):
        """
        Detect persons and faces in the image
        Args:
            image: input image(s)
            
        Returns:
            PersonAndFaceResult: Object containing detection results
        """
        with torch.no_grad():
            results = self.yolo(
                source=images,
                conf=0.5,
                classes=[0, 1],  # 0: person, 1: face
                device=self.device,
                verbose=False
            )
            return PersonAndFaceResult(results[0])
    
    # Add an alias for the predict method
    detect = predict
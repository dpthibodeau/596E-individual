import torch
from ultralytics import YOLO
import os
from pathlib import Path
import shutil

def export_yolo_to_onnx(model_path: str, output_path: str):
    """Convert YOLO model to ONNX format"""
    print(f"Converting YOLO model from {model_path} to ONNX...")
    
    _torch_load_original = torch.load
    
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _torch_load_original(*args, **kwargs)
    
    torch.load = safe_load
    
    try:
        # Load YOLO model
        model = YOLO(model_path)
        
        # Export to ONNX
        success = model.export(format='onnx',
                             opset=12,
                             simplify=True,
                             dynamic=True,
                             imgsz=640)  # Use default YOLO image size
        
        if success:
            default_onnx_path = str(Path(model_path).with_suffix('.onnx'))
            if os.path.exists(default_onnx_path):
                shutil.move(default_onnx_path, output_path)
            print(f"YOLO model exported successfully to: {output_path}")
            
            # Verify the exported model
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification successful")
        else:
            print("YOLO model export failed")
            
    except Exception as e:
        print(f"Error during YOLO model export: {str(e)}")
        raise
        
    finally:
        torch.load = _torch_load_original
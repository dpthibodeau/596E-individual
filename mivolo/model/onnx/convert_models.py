import os
from mivolo.model.onnx.yolo_converter import export_yolo_to_onnx
from mivolo.model.onnx.mivolo_converter import export_mivolo_to_onnx

def convert_models():
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # Create output directory
    onnx_dir = os.path.join(project_root, "models", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    
    # Convert YOLO model
    yolo_path = os.path.join(project_root, "models", "yolov8x_person_face.pt")
    yolo_onnx_path = os.path.join(onnx_dir, "yolov8x_person_face.onnx")
    export_yolo_to_onnx(yolo_path, yolo_onnx_path)
    
    # Convert MiVOLO model
    mivolo_path = os.path.join(project_root, "models", "mivolo_imdb.pt.tar")
    mivolo_onnx_path = os.path.join(onnx_dir, "mivolo_imdb.onnx")
    export_mivolo_to_onnx(mivolo_path, mivolo_onnx_path)

if __name__ == "__main__":
    convert_models()
import torch
from dataclasses import dataclass

@dataclass
class Boxes:
    xyxy: torch.Tensor

@dataclass
class Results:
    boxes: Boxes

@dataclass
class DetectedObjects:
    yolo_results: Results
    ages: list
    genders: list
    face_to_person_map: dict
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict

class PersonAndFaceCrops:
    def __init__(self, face_crops: List[np.ndarray], body_crops: List[np.ndarray], 
                 face_indices: List[int], body_indices: List[int]):
        self.face_crops = face_crops
        self.body_crops = body_crops
        self.face_indices = face_indices
        self.body_indices = body_indices

    def get_faces_with_bodies(self, use_person_crops: bool, use_face_crops: bool) -> Tuple[Tuple[List[int], List[np.ndarray]], Tuple[List[int], List[np.ndarray]]]:
        """Return faces and bodies that are associated with each other"""
        if not use_face_crops:
            faces = [None] * len(self.face_indices)
        else:
            faces = self.face_crops
            
        if not use_person_crops:
            bodies = [None] * len(self.body_indices)
        else:
            bodies = self.body_crops
            
        return (self.body_indices, bodies), (self.face_indices, faces)

class PersonAndFaceResult:
    def __init__(self, yolo_results):
        self.yolo_results = yolo_results
        self.orig_img = yolo_results.orig_img
        self.n_objects = len(yolo_results.boxes)
        self.boxes = yolo_results.boxes.xyxy.cpu().numpy() if len(yolo_results.boxes) > 0 else []
        self.classes = yolo_results.boxes.cls.cpu().numpy() if len(yolo_results.boxes) > 0 else []
        self.conf = yolo_results.boxes.conf.cpu().numpy() if len(yolo_results.boxes) > 0 else []
        self.face_to_person_map: Dict[int, int] = {}
        self.ages: Dict[int, Optional[float]] = {}
        self.genders: Dict[int, Optional[str]] = {}
        self.gender_scores: Dict[int, Optional[float]] = {}
        
        # Count persons and faces
        self.n_persons = sum(1 for cls in self.classes if cls == 0)
        self.n_faces = sum(1 for cls in self.classes if cls == 1)

    def associate_faces_with_persons(self):
        """Associate detected faces with person detections"""
        if len(self.boxes) == 0:
            return
            
        person_indices = np.where(self.classes == 0)[0]
        face_indices = np.where(self.classes == 1)[0]
        
        for face_idx in face_indices:
            face_box = self.boxes[face_idx]
            face_center = [(face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2]
            
            for person_idx in person_indices:
                person_box = self.boxes[person_idx]
                if (face_center[0] >= person_box[0] and 
                    face_center[0] <= person_box[2] and 
                    face_center[1] >= person_box[1] and 
                    face_center[1] <= person_box[3]):
                    self.face_to_person_map[face_idx] = person_idx
                    break

    def collect_crops(self, image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> 'PersonAndFaceCrops':
        """Extract crops of faces and bodies"""
        face_crops = []
        body_crops = []
        face_indices = []
        body_indices = []
        
        # Get face indices with associated bodies
        faces_with_bodies = list(self.face_to_person_map.keys())
        
        for i, box in enumerate(self.boxes):
            crop = None
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Add padding
            h, w = image.shape[:2]
            pad = max(int((x2 - x1) * 0.1), int((y2 - y1) * 0.1))
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            if self.classes[i] == 1 and i in faces_with_bodies:  # Face
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, size)
                    face_crops.append(crop)
                    face_indices.append(i)
                    
                    # Get corresponding body crop
                    person_idx = self.face_to_person_map[i]
                    px1, py1, px2, py2 = map(int, self.boxes[person_idx])
                    body_crop = image[py1:py2, px1:px2]
                    if body_crop.size > 0:
                        body_crop = cv2.resize(body_crop, size)
                        body_crops.append(body_crop)
                        body_indices.append(person_idx)
        
        return PersonAndFaceCrops(face_crops, body_crops, face_indices, body_indices)

    def set_age(self, index: int, age: float):
        """Set age for a detection"""
        self.ages[index] = age

    def set_gender(self, index: int, gender: str, score: float = None):
        """Set gender for a detection"""
        self.genders[index] = gender
        if score is not None:
            self.gender_scores[index] = score

    def plot(self, image: Optional[np.ndarray] = None, show_conf: bool = True) -> np.ndarray:
        """Draw all detections on the image"""
        if image is None:
            image = self.orig_img.copy()
        else:
            image = image.copy()

        # Colors for different classes (person: blue, face: green)
        colors = {
            0: (255, 0, 0),  # person - blue
            1: (0, 255, 0)   # face - green
        }

        thickness = max(round(sum(image.shape) / 2000), 1)
        font_thickness = max(thickness - 1, 1)
        font_scale = thickness * 0.5

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cls = int(self.classes[i])
            conf = self.conf[i]

            # Draw box
            color = colors.get(cls, (0, 0, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Prepare label
            label_parts = []
            label_parts.append('person' if cls == 0 else 'face')
            if show_conf:
                label_parts.append(f'{conf:.2f}')
            if i in self.ages:
                label_parts.append(f'age:{self.ages[i]:.1f}')
            if i in self.genders:
                label_parts.append(self.genders[i])
                if i in self.gender_scores:
                    label_parts.append(f'{self.gender_scores[i]:.2f}')
            
            label = ' '.join(label_parts)

            # Draw label background
            text_size = cv2.getTextSize(label, 0, font_scale, font_thickness)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 3), 
                         (x1 + text_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(image, label, (x1, y1 - 2), 0, font_scale,
                       (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

        return image
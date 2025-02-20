import onnxruntime as ort
import numpy as np
import cv2
import torch
from typing import List, Tuple, Dict
from .utils import DetectedObjects, Results, Boxes
from sklearn.cluster import DBSCAN

class ONNXPreProcessor:
    def __init__(self):
        self.yolo_size = (640, 640)
        self.mivolo_size = (224, 224)
        
    def prepare_yolo_input(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.yolo_size)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = np.ascontiguousarray(img.astype(np.float32) / 255.0)
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, 0)

    def prepare_mivolo_input(self, img):
        img = cv2.resize(img, self.mivolo_size)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        cl = clahe.apply(l)
        img = cv2.merge((cl,a,b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        img = np.ascontiguousarray(img.astype(np.float32) / 255.0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return np.concatenate([img, img], axis=1)

class ONNXInference:
    def __init__(self, yolo_path: str, mivolo_path: str):
        self.yolo_session = ort.InferenceSession(yolo_path)
        self.mivolo_session = ort.InferenceSession(mivolo_path)
        self.preprocessor = ONNXPreProcessor()
        self.confidence_threshold = 0.02
        self.min_face_size = 20
        self.max_face_size = 800
        self.scale_factors = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5]

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def nms_with_overlap_threshold(self, boxes, scores, num_faces=1):
        if len(boxes) == 0:
            return []
                
        iou_threshold = 0.3
        overlap_threshold = 0.2
                
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
                    
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
                
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
                
            intersection = w * h
            union = areas[i] + areas[indices[1:]] - intersection
            iou = intersection / union
            overlap_ratios = intersection / np.minimum(areas[i], areas[indices[1:]])
                
            keep_mask = (iou <= iou_threshold) & (overlap_ratios <= overlap_threshold)
            indices = indices[1:][keep_mask]
        
        return keep

    def preprocess_image(self, img, max_size=2400):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        img = cv2.merge((cl,a,b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        gamma = 1.1
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, lookup_table)
        
        orig_height, orig_width = img.shape[:2]
        scale = min(max_size / max(orig_height, orig_width), 1.0)
        if scale < 1.0:
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return img, scale

    def process_yolo_scale(self, img, scale_factor=1.0):
        img_height, img_width = img.shape[:2]
        
        if scale_factor != 1.0:
            new_height = int(img_height * scale_factor)
            new_width = int(img_width * scale_factor)
            img = cv2.resize(img, (new_width, new_height))
        
        yolo_input = self.preprocessor.prepare_yolo_input(img)
        outputs = self.yolo_session.run(None, {"images": yolo_input})
        detection = outputs[0][0].transpose()
        
        boxes = []
        scores = []
        
        for i in range(detection.shape[0]):
            confidence = float(detection[i, 4])
            if confidence > self.confidence_threshold:
                cx = float(detection[i, 0]) * img_width / 640
                cy = float(detection[i, 1]) * img_height / 640
                
                face_width = float(detection[i, 2]) * img_width / 640
                face_height = float(detection[i, 3]) * img_height / 640
                
                face_size = max(face_width, face_height)
                face_size = max(face_size, self.min_face_size)
                
                padding = face_size * 0.2
                half_size = (face_size + padding) / 2
                
                x1 = max(0, min(img_width, cx - half_size))
                x2 = max(0, min(img_width, cx + half_size))
                y1 = max(0, min(img_height, cy - half_size))
                y2 = max(0, min(img_height, cy + half_size))
                
                if scale_factor != 1.0:
                    x1 /= scale_factor
                    x2 /= scale_factor
                    y1 /= scale_factor
                    y2 /= scale_factor
                
                x1 = max(0, min(img_width, x1))
                x2 = max(0, min(img_width, x2))
                y1 = max(0, min(img_height, y1))
                y2 = max(0, min(img_height, y2))
                
                width = x2 - x1
                height = y2 - y1
                area_ratio = (width * height) / (img_width * img_height)
                
                if (width >= self.min_face_size and 
                    height >= self.min_face_size and
                    width <= self.max_face_size and
                    height <= self.max_face_size and
                    0.000005 < area_ratio < 0.3 and
                    abs(width - height) / max(width, height) < 0.3):
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)
        
        return boxes, scores

    def validate_box(self, box, img_height, img_width):
        x1, y1, x2, y2 = box
        x1 = max(0, min(img_width, x1))
        x2 = max(0, min(img_width, x2))
        y1 = max(0, min(img_height, y1))
        y2 = max(0, min(img_height, y2))
        
        width = x2 - x1
        height = y2 - y1
        
        if width < self.min_face_size or height < self.min_face_size:
            return None
        if width > self.max_face_size or height > self.max_face_size:
            return None
            
        return [x1, y1, x2, y2]

    def recognize(self, img):
        processed_img, scale = self.preprocess_image(img)
        all_boxes = []
        all_scores = []
        
        for scale_factor in self.scale_factors:
            boxes, scores = self.process_yolo_scale(processed_img, scale_factor)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
        
        if not all_boxes:
            return DetectedObjects(Results(Boxes(xyxy=torch.tensor([]))), [], [], {}), img
        
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        
        if len(boxes) > 2:
            face_centers = np.array([(box[0] + box[2])/2 for box in boxes])
            clustering = DBSCAN(eps=0.25, min_samples=1).fit(face_centers.reshape(-1, 1))
            num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        else:
            num_clusters = 1
            
        keep_indices = self.nms_with_overlap_threshold(boxes, scores, num_faces=num_clusters)
        boxes = boxes[keep_indices]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        ages = []
        genders = []
        raw_ages = []
        face_sizes = []
        face_to_person_map = {}
        valid_boxes = []
        
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = map(int, box[:4])
                padding_x = int((x2 - x1) * 0.1)
                padding_y = int((y2 - y1) * 0.1)
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(img.shape[1], x2 + padding_x)
                y2 = min(img.shape[0], y2 + padding_y)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                    
                mivolo_input = self.preprocessor.prepare_mivolo_input(face)
                mivolo_outputs = self.mivolo_session.run(None, {"input": mivolo_input})
                pred = mivolo_outputs[0][0]
                
                raw_age = float(pred[0])
                face_size = min(x2 - x1, y2 - y1)
                
                raw_ages.append(raw_age)
                face_sizes.append(face_size)
                face_to_person_map[len(valid_boxes)] = len(valid_boxes)
                valid_boxes.append(box.tolist())
                
            except Exception as e:
                print(f"Error processing box {i}: {e}")
                continue
        
        if not valid_boxes:
            return DetectedObjects(Results(Boxes(xyxy=torch.tensor([]))), [], [], {}), img
        
        max_face_size = max(face_sizes)
        normalized_ages = []
        for raw_age, face_size in zip(raw_ages, face_sizes):
            size_factor = face_size / max_face_size
            base_age = 1 / (1 + np.exp(-raw_age)) * 70  
            
            if size_factor < 0.7:  
                adjusted_age = base_age * 0.7
            else:
                adjusted_age = base_age
                
            normalized_ages.append(adjusted_age)
        
        sorted_ages = sorted(normalized_ages)
        age_gaps = [sorted_ages[i+1] - sorted_ages[i] for i in range(len(sorted_ages)-1)]
        
        if len(age_gaps) > 0:
            median_gap = np.median(age_gaps)
            for i in range(len(normalized_ages)):
                if i > 0 and normalized_ages[i] - normalized_ages[i-1] < median_gap/2:
                    normalized_ages[i] = normalized_ages[i-1] + median_gap/2
        
        valid_boxes = torch.tensor(valid_boxes, dtype=torch.float32)
        
        # Process each face with normalized ages
        for i, box in enumerate(valid_boxes):
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            mivolo_input = self.preprocessor.prepare_mivolo_input(face)
            mivolo_outputs = self.mivolo_session.run(None, {"input": mivolo_input})
            pred = mivolo_outputs[0][0]
            
            size_factor = face_sizes[i] / max_face_size
            base_threshold = 0.1
            gender_threshold = base_threshold + (0.1 * (1 - size_factor))  
            
            gender = "female" if pred[1] > gender_threshold else "male"
            label = "child" if normalized_ages[i] <= 22 else "adult"
            
            ages.append(normalized_ages[i])
            genders.append(gender)
        
        detected_objects = DetectedObjects(
            yolo_results=Results(boxes=Boxes(xyxy=valid_boxes)),
            ages=ages,
            genders=genders,
            face_to_person_map=face_to_person_map
        )
        
        output_img = img.copy()
        for i, box in enumerate(valid_boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{ages[i]:.1f}, {genders[i]}"
            cv2.putText(output_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detected_objects, output_img
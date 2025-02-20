import argparse
import os
import json
import csv
from typing import List, Dict
import cv2
from mivolo.model.onnx.onnx_inference import ONNXInference
import torch


def get_images(folder_dir: str) -> List[str]:
    """Get all image files from directory"""
    images = []
    for image in os.listdir(folder_dir):
        if image.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(folder_dir, image))
    return images


def process_directory(
    input_dir: str,
    output_dir: str,
    single_person: bool = False,
    store_images: bool = True,
) -> Dict:
    """Process all images in a directory and return results"""
    # Initialize predictor
    predictor = ONNXInference(
        yolo_path="models/onnx/yolov8x_person_face.onnx",
        mivolo_path="models/onnx/mivolo_imdb.onnx",
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if store_images:
        output_images_dir = os.path.join(output_dir, "processed_images")
        os.makedirs(output_images_dir, exist_ok=True)

    # Get input images
    images = get_images(input_dir)
    if not images:
        raise ValueError(f"No valid images found in directory: {input_dir}")

    # Process images
    results = []
    for image_path in images:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Get predictions
            detected_objects, output_img = predictor.recognize(img)

            # Save processed image if requested
            if store_images:
                output_path = os.path.join(
                    output_images_dir, os.path.basename(image_path)
                )
                cv2.imwrite(output_path, output_img)

            # Process detections
            image_results = process_detections(
                detected_objects, single_person, image_path
            )
            results.append(image_results)

            print(f"Processed: {image_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results.append({"file_path": image_path, "result": f"Error: {str(e)}"})

    return results


def process_detections(detected_objects, single_person: bool, image_path: str) -> Dict:
    """Process detected objects for a single image"""
    bboxes = detected_objects.yolo_results.boxes.xyxy.cpu().numpy()
    ages = detected_objects.ages
    genders = detected_objects.genders
    face_indexes = detected_objects.face_to_person_map.keys()

    detections = []
    avg_age = 0

    for i in face_indexes:
        if ages[i] is not None:
            avg_age += ages[i]
            detections.append(
                {
                    "bbox": {
                        "X1": int(bboxes[i][0]),
                        "Y1": int(bboxes[i][1]),
                        "X2": int(bboxes[i][2]),
                        "Y2": int(bboxes[i][3]),
                    },
                    "label": "child" if int(ages[i]) <= 22 else "adult",
                    "gender": genders[i],
                }
            )

    if not detections:
        return {"file_path": image_path, "result": "No person detected"}

    if single_person and detections:
        avg_age = avg_age / len(face_indexes)
        return {
            "file_path": image_path,
            "result": [
                {**detections[0], "label": "child" if int(avg_age) <= 22 else "adult"}
            ],
        }

    return {"file_path": image_path, "result": detections}


def save_results(results: List[Dict], output_dir: str):
    """Save results to JSON and CSV files"""
    # Save JSON results
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved JSON results to: {json_path}")

    # Generate CSV summary
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file_path",
                "Num of detected females",
                "Num of detected males",
                "Num of detected children",
                "Num of detected adults",
            ]
        )

        for img in results:
            row = [img["file_path"]]
            if isinstance(img["result"], str):
                row.extend([0, 0, 0, 0])
            else:
                nf = nm = nc = na = 0
                for person in img["result"]:
                    if person["label"] == "child":
                        nc += 1
                    else:
                        na += 1
                    if person["gender"] == "male":
                        nm += 1
                    else:
                        nf += 1
                row.extend([nf, nm, nc, na])
            writer.writerow(row)

    print(f"Saved CSV summary to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Age and Gender Classifier CLI")
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output files"
    )
    parser.add_argument(
        "--single-person",
        action="store_true",
        help="Only process the first detected person in each image",
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Don't save processed images"
    )

    args = parser.parse_args()

    # Enable CUDA optimizations if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("CUDA optimizations enabled")

    try:
        results = process_directory(
            args.input_dir, args.output_dir, args.single_person, not args.no_images
        )
        save_results(results, args.output_dir)
        print("Processing completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

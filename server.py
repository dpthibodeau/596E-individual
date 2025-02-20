import argparse
import os
import json
import csv
from typing import TypedDict
from mivolo.model.onnx.onnx_inference import ONNXInference
import cv2
import torch
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    EnumParameterDescriptor,
    EnumVal,
    FileResponse,
    InputSchema,
    InputType,
    ParameterSchema,
    ResponseBody,
    TaskSchema,
    BatchFileResponse,
)
from mivolo.predictor import Predictor


def get_images(folder_dir):
    """Get all image files from directory"""
    images = []
    for image in os.listdir(folder_dir):
        if image.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(folder_dir, image))
    return images


def get_parser():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector-weights",
        default=os.path.join(current_dir, "models", "yolov8x_person_face.pt"),
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(current_dir, "models", "mivolo_imdb.pth.tar"),
        type=str,
    )
    parser.add_argument("--with-persons", action="store_false")
    parser.add_argument("--disable_faces", action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--single-person", action="store_true")
    parser.add_argument("--draw", action="store_false")
    parser.add_argument("--port", default=5000, type=int)
    return parser


def generate_img_data(imgs):
    """Generate CSV data from image results"""
    data = []
    for img in imgs:
        img_data = []
        img_data.append(img["file_path"])
        nf = nm = nc = na = 0
        if isinstance(img["result"], str):  # Handle "No person detected" case
            img_data.extend([0, 0, 0, 0])
        else:
            for person in img["result"]:
                if person["label"] == "child":
                    nc += 1
                else:
                    na += 1
                if person["gender"] == "male":
                    nm += 1
                else:
                    nf += 1
            img_data.extend([nf, nm, nc, na])
        data.append(img_data)
    return data


def classify_given_age(age):
    """Classify age into child/adult categories"""
    return "child" if age <= 22 else "adult"


def create_transform_case_task_schema() -> TaskSchema:
    """Create the task schema for the API endpoint"""
    input_schema = InputSchema(
        key="input_directory",
        label="Path to the directory containing all the images",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_directory",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )
    single_person_schema = ParameterSchema(
        key="single_person",
        label="Single person flag",
        value=EnumParameterDescriptor(
            default="False",
            enum_vals=[
                EnumVal(key="True", label="True"),
                EnumVal(key="False", label="False"),
            ],
        ),
    )
    store_images_schema = ParameterSchema(
        key="store_images",
        label="Store images",
        value=EnumParameterDescriptor(
            default="True",
            enum_vals=[
                EnumVal(key="True", label="True"),
                EnumVal(key="False", label="False"),
            ],
        ),
    )
    return TaskSchema(
        inputs=[input_schema, output_schema],
        parameters=[single_person_schema, store_images_schema],
    )


class Inputs(TypedDict):
    input_directory: DirectoryInput
    output_directory: DirectoryInput


class Params(TypedDict):
    single_person: str
    store_images: str


parser = get_parser()
params = parser.parse_args()

# Verify model files exist
for model_path in [params.detector_weights, params.checkpoint]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Please download the required models first."
        )

predictor = ONNXInference(
    yolo_path="models/onnx/yolov8x_person_face.onnx",
    mivolo_path="models/onnx/mivolo_imdb.onnx",
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

server = MLServer(__name__)
server.add_app_metadata(
    name="Age and Gender Classifier",
    author="User",
    version="1.0.0",
    info=load_file_as_string("README.md"),
)


@server.route(
    "/classify_age_gender", task_schema_func=create_transform_case_task_schema
)
def classify(inputs: Inputs, parameters: Params) -> ResponseBody:
    """Process images and classify age/gender"""
    input_folder_dir = inputs["input_directory"].path
    output_folder_dir = inputs["output_directory"].path
    hash_name = str(torch.randint(0, 1000000000, (1,)).item())

    # Get input images
    images = get_images(input_folder_dir)
    if not images:
        raise ValueError("No valid images found in input directory")

    # Process parameters
    single_person_flag = parameters["single_person"].lower() == "true"
    store_images = parameters["store_images"].lower() == "true"

    # Create output directory if storing images
    if store_images:
        output_images_path = os.path.join(output_folder_dir, f"outputs_{hash_name}")
        os.makedirs(output_images_path, exist_ok=True)

    # Process each image
    main_res = []
    no_predict = 0

    for image_name in images:
        try:
            avg_age = 0
            res = []

            # Read image
            img = cv2.imread(image_name)
            if img is None:
                raise ValueError(f"Failed to load image: {image_name}")

            # Get predictions
            detected_objects, output_img = predictor.recognize(img)

            # Save annotated image if requested
            if store_images:
                cv2.imwrite(
                    os.path.join(output_images_path, os.path.basename(image_name)),
                    output_img,
                )

            # Process detections
            bboxes = detected_objects.yolo_results.boxes.xyxy.cpu().numpy()
            ages = detected_objects.ages
            genders = detected_objects.genders
            face_indexes = detected_objects.face_to_person_map.keys()

            # Process each detected face
            for i in face_indexes:
                if ages[i] is not None:
                    avg_age += ages[i]
                    res.append(
                        {
                            "bbox": {
                                "X1": int(bboxes[i][0]),
                                "Y1": int(bboxes[i][1]),
                                "X2": int(bboxes[i][2]),
                                "Y2": int(bboxes[i][3]),
                            },
                            "label": classify_given_age(int(ages[i])),
                            "gender": genders[i],
                        }
                    )

            # Handle no detections
            if not res:
                no_predict += 1
                main_res.append(
                    {"file_path": image_name, "result": "No person detected"}
                )
                continue

            # Handle single person mode
            if single_person_flag and res:
                res = [res[0]]
                res[0]["label"] = classify_given_age(int(avg_age / len(face_indexes)))

            main_res.append({"file_path": image_name, "result": res})

        except Exception as e:
            print(f"Error processing image {image_name}: {str(e)}")
            main_res.append({"file_path": image_name, "result": f"Error: {str(e)}"})

    # Save results to JSON
    result_path = os.path.join(output_folder_dir, f"{hash_name}_result.json")
    with open(result_path, "w") as f:
        json.dump(main_res, f, indent=4)

    # Generate and save CSV summary
    img_data_dict = generate_img_data(main_res)
    csv_path = os.path.join(output_folder_dir, f"{hash_name}_csv_info.csv")
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
        writer.writerows(img_data_dict)

    res_body = [
        FileResponse(path=result_path, file_type="json"),
        FileResponse(path=csv_path, file_type="csv"),
    ]
    return ResponseBody(BatchFileResponse(files=res_body))


if __name__ == "__main__":
    server.run(port=params.port)

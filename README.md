# Classifying images based on Age/Gender

** Modified from https://github.com/pritshah9/age_classification_596E **

This project is developed as part of 596E course at UMass Amherst. It works with the RescueBox app. It uses the [MiVolo](https://github.com/WildChlamydia/MiVOLO) model for bounding box predictions on images and then to classify whether the person(s) is a child or an adult and classifies their gender too.

IMPORTANT: Note that all results computed by the program are **predictions**, they may be incorrect or misleading.

**Setting up the project and exporting to ONNX**

1. Download the files in this repo into a folder (git clone or direct download).
2. Create a new subfolder called `models`. Your structure should be like so -

```
596E-individual\
│
├── mivolo\
├── models\
├── server.py
├── requirements.txt
...
```

4. [Download](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view) body + face detector model to `models/yolov8x_person_face.pt`
5. [Download](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) mivolo checkpoint and rename to `models/mivolo_imdb.pth.tar`
6. install packages on machine
`brew install cmake`<br>
`brew install protobuf`
7. Create a conda env with `conda create -n <my_env_name> python=3.10` then activate with `conda activate <my_env_name>`
8. install conda packages:
`conda install -c conda-forge onnx==1.14.0`<br>
`conda install -c conda-forge onnxruntime`<br>
`conda install pytorch torchvision -c pytorch`<br>
`conda install -c conda-forge protobuf`
9. Open a terminal window, navigate to this folder and run `pip install -r "requirements.txt"` to download the libraries required.
10. Convert the models to ONNX format using script `python -m mivolo.model.onnx.convert_models`<br>
    or `<path_to_correct_python_conda_installation>/python -m mivolo.model.onnx.convert_models`<br>
    (i.e. `/Users/davidthibodeau/miniconda3/envs/myenv/bin/python -m mivolo.model.onnx.convert_models`)

**Basic Usage**

1. In the command line, run `python server.py` to run the server. By default the server will run on port 5000. To specify another port, use the following command - `python server.py --port <your_port_num>`. You may have to use `<path_to_correct_python_conda_installation>/python server.py` (i.e. `/Users/davidthibodeau/miniconda3/envs/myenv/bin/python server.py`), as Python 3.10 was used in this project.
   
2. The RescueBox application is the easiest way to use the model and can be found [here](https://github.com/UMass-Rescue/RescueBox-Desktop). Instructions on how to use it and register our server are present in the link given.

3. There is also a CLI for running this model straight from a terminal session. From the project directory, you may run cli.py. On my machine, I included the direct path to the correct python installation: `<path_to_correct_python_conda_installation>/python cli.py`. Arguments to the CLI are listed below:
   --input-dir: Directory containing your input images (required)<br>
   --output-dir: Directory where results will be saved (required)<br>
   --single-person: Only process the first detected person in each image (optional)<br>
   --no-images: Don't save the processed images with annotations (optional)

Example usage of the CLI using the included test images and saving to an outputs folder:
`/Users/davidthibodeau/miniconda3/envs/myenv/bin/python cli.py --input-dir ./test_data --output-dir ./outputs`

<img width="708" alt="Screenshot 2025-02-19 at 11 02 54 AM" src="https://github.com/user-attachments/assets/9b467746-957b-4a5e-9e0f-e319058bc17b" />


5. Once configured, this model will need the path to the directory containing input images. It will also need a path to where the output should be stored.

6. Two optional arguments are also available. If **Store images** is true, the program will store annotated (age, gender) images as part of an folder starting with `outputs_` which will be created at the output path. The output images will have boxes around the detected faces. I have labeled every face with detected age under 22 as a child (to minimize the chances of mislabeling a child). If **Single person flag** is set to be true, the program will assume each image only has a single face and will give one prediction per image. 

7. There will be a csv and a json output file. The json file is a complete overview, each item in the main list corresponds to an image and has the image path and results. The results will be a list of objects where each object corresponds to a face and has the gender, age and bounding box predictions. The csv gives a more condensed analysis and includes 4 attributes per image - number of children, adults, male persons and female persons. Note again these are just **predictions** and may be incorrect!

**Optional CLI arguments (before running server)**

- `--device <device>` may be used to specify a different device to run the model on. Default is `cpu`.
- `--checkpoint <checkpoint>` may be used to specify a different mivolo model checkpoint (.pth.tar file) to use for classification.
- `--detector_weights <checkpoint>` may be used to specify a different yolo bounding box prediction model (.pt file) to use for classification.


**Future work**

- More rigorous evaluation of the model.
- Come up with a better way of presenting results to the users.
- Extending usage to videos.


## Conversion to ONNX runtime

The convert_models.py file located in /mivolo/model/onnx handles the conversion of both the Yolo and MiVolo models used in this project. convert_models.py calls on mivolo_converter.py and yolo_converter.py to pre-process and convert the models according to their use. In those files, the export commands can be found in those files:

<img width="500" alt="Screenshot 2025-02-16 at 2 51 06 PM" src="https://github.com/user-attachments/assets/0b482d4c-13b0-49ea-b0ff-624bdf8659f9" /> <br>
and <br>
<img width="500" alt="Screenshot 2025-02-16 at 2 49 50 PM" src="https://github.com/user-attachments/assets/f0409ad7-b079-405e-a986-1d99bb76533c" /> <br>
<img width="500" alt="Screenshot 2025-02-16 at 2 50 14 PM" src="https://github.com/user-attachments/assets/c8ff9d78-9674-45d7-a6a0-f397af2692cb" /> <br>

After convert_models.py is run per the instructions above, the /models/onnx directory is created and the .onnx models are stored there. They are used in the code as seen in server.py:

<img width="499" alt="Screenshot 2025-02-17 at 5 41 06 PM" src="https://github.com/user-attachments/assets/a4ea961c-f489-4874-bad8-2f0345c14c5f" /> <br>
<img width="540" alt="Screenshot 2025-02-17 at 5 42 20 PM" src="https://github.com/user-attachments/assets/469ece82-9460-4915-911c-8036ae06bb67" /> <br>

**Example RescueBox Output:**


<img width="1022" alt="Screenshot 2025-02-19 at 7 40 57 PM" src="https://github.com/user-attachments/assets/e04ba7e2-ae96-43de-979e-73f1265f48fc" />

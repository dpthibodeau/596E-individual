# Classifying images based on Age/Gender

This project is developed as part of 596E course at UMass Amherst. It works with the RescueBox app. It uses the [MiVolo](https://github.com/WildChlamydia/MiVOLO) model for bounding box predictions on images and then to classify whether the person(s) is a child or an adult and classifies their gender too.

IMPORTANT: Note that all results computed by the program are **predictions**, they may be incorrect or misleading.

**Requirements**

1. Download the files in this repo into a folder (git clone or direct download).
2. Download the `mivolo` folder on [this](https://github.com/WildChlamydia/MiVOLO) repo. Use the following link to quick download just the folder - [link](https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2FWildChlamydia%2FMiVOLO%2Ftree%2Fmain%2Fmivolo)
3. Create a new subfolder called `models`. Your structure should be like so -

```
age_gender_detect\
│
├── mivolo\
├── models\
├── server.py
├── requirements.txt
...
```

4. [Download](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view) body + face detector model to `models/yolov8x_person_face.pt`
5. [Download](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) mivolo checkpoint to `models/mivolo_imbd.pth.tar`
6. Open a terminal window, navigate to this folder and run `pip install -r "requirements.txt"` to download the libraries required. 

**Basic Usage**

1. In the command line, run `python3 server.py` to run the server. By default the server will run on port 5000. To specify another port, use the following command - `python3 server.py --port <your_port_num>`
2. The RescueBox application is needed to use the model and can be found [here](https://github.com/UMass-Rescue/RescueBox-Desktop). Instructions on how to use it and register our server are present in the link given.

3. Once configured, this model will need the path to the directory containing input images. It will also need a path to where the output should be stored.

4. Two optional arguments are also available. If **Store images** is true, the program will store annotated (age, gender) images as part of an folder starting with `outputs_` which will be created at the output path. The output images will have boxes around the detected faces. I have labeled every face with detected age under 22 as a child (to minimize the chances of mislabeling a child). If **Single person flag** is set to be true, the program will assume each image only has a single face and will give one prediction per image. 

5. There will be a csv and a json output file. The json file is a complete overview, each item in the main list corresponds to an image and has the image path and results. The results will be a list of objects where each object corresponds to a face and has the gender, age and bounding box predictions. The csv gives a more condensed analysis and includes 4 attributes per image - number of children, adults, male persons and female persons. Note again these are just **predictions** and may be incorrect!

**Optional CLI arguments (before running server)**

- `--device <device>` may be used to specify a different device to run the model on. Default is `cuda:0`.
- `--checkpoint <checkpoint>` may be used to specify a different mivolo model checkpoint (.pth.tar file) to use for classification.
- `--detector_weights <checkpoint>` may be used to specify a different yolo bounding box prediction model (.pt file) to use for classification.


**Future work**

- More rigorous evaluation of the model.
- Come up with a better way of presenting results to the users.
- Extending usage to videos.
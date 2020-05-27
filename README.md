# Face Recognition System

## 1. Face Recognition System

First of all, face images we want to recognize need to be collected.

### Face Registration Phase

Face images must be gathered under following conditions to generalize the model better.

- Different lighting conditions
- Time of the day (day or night)
- Different face orientation
- Different face expressions if possible (e.g. yawn, stretch, etc..)

In my script, only images are saved when the face is clearly detected by MTCNN to remove any unnecessary data.

```
python3 face_register_gui.py
```

All the saved images are stored in the directory with the person name under "dataset" directory.

### Face Recognition Phase

In this phase, I will use the face images collected from the registration stage to build the Face Recognizer Model.


**Step-by-step Training Procedure**

- detect an input image with ***MTCNN*** to detect whether or not face is present
- extract face features from the image with ***InceptionResnetV1*** (pretrained on 'VGGFace2')
- use ***LogisticRegression*** to classify the faces

Default options are:

- type - webcam
- path - 0
 
```
python3 face_recognizer.py --type [webcam, video, image] --path <path to file> 
```

If you want to save the output video, '--save' option is required.

## 2. Person Detection System

For person detection, I used **ChainerCV** library with **yolo_v3** pretrained model.

Currently, it will draw bounding box for one person only.

When two or more person are detected, it will display a text.

# Requirements

```
pip3 install chainer chainercv
```

# Usage

```
python3 detect_person.py
```


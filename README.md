
# Smart Wheelchair Control System Based on SIBI Gesture and Smart Braking System using YOLOv8 and LSTM

This time the system will use two cameras to implement the control and smart braking features for wheelchairs, the first camera is specifically for detecting obstacles with the YOLOv8 model that has been trained, where if the obstacle is below 1.2 meters it will send a stop command. The second camera is used to control the wheelchair using LSTM with Sibi language gestures.

![Bisindo](https://github.com/user-attachments/assets/5441117f-9c74-45e2-a3b8-1f6f32690baa)

## 🎬 Demo

![demo](https://github.com/user-attachments/assets/6df1c74a-0625-4650-8092-10bbafc2a47d)


## 🔨 Installation

Pypi version

![MediaPipe version](https://img.shields.io/badge/MediaPipe-v0.10.14-blue)
![Ultralytics version](https://img.shields.io/badge/Ultralytics-v8.1.42-pink)
![Tensorflow version](https://img.shields.io/badge/Tensorflow-v2.10.1-orange)
![OpenCV version](https://img.shields.io/badge/OpenCV-v4.9.0.80-green)
![IPyKernel version](https://img.shields.io/badge/IPyKernel-v6.29.4-yellow)
![Scikit-learn version](https://img.shields.io/badge/scikitlearn-v1.5.1-black)
![Keras version](https://img.shields.io/badge/Keras-v3.5.0-purple)
![matplotlib version](https://img.shields.io/badge/matplotlib-v3.9.2-red)
![MediaPipe version](https://img.shields.io/badge/MediaPipe-v0.10.14-blue)


I recommend a separate folder and venv for creating the model and the wheelchair program. for training venv setup :

for training LSTM model
```bash
  python --version
  python -m venv nama_venv
  nama_venv\Scripts\activate
  pip install opencv-python
  pip install mediapipe
  pip install numpy
  pip install matplotlib
  pip install tensorflow
  pip install seaborn
  pip install scikit-learn
```
for training YOLOv8
```bash
  pip install ipykernel
  pip install ultralytics roboflow opencv-python
```

YOLOv8 need an absolute path. so change the data path for train, val, test in data.yaml example :

```bash
  names:
  -Manusia
  nc: 1
  roboflow:
    license: CC BY 4.0
    project: deteksi-manusia-yolov8-dataset
    url: https://universe.roboflow.com/hari-vijaya-kusuma/deteksi-manusia-yolov8-dataset/dataset/1
    version: 1
    workspace: hari-vijaya-kusuma
  test: D:/train yolo/Deteksi-Manusia-YoloV8-Dataset-1/test/images
  train: D:/train yolo/Deteksi-Manusia-YoloV8-Dataset-1/train/images
  val: D:/train yolo/Deteksi-Manusia-YoloV8-Dataset-1/valid/images
```

If you feel the training is taking too long, but you have a computer with NASA Super Computer like specs, it means you haven't utilized your GPU for the training process. check for cuda by running this

```bash
  import torch
  print(torch.cuda.is_available())
```

if the output is false then you need to install PyTorch with CUDA support. Check your driver version before implementing CUDA. example for installing CUDA 11.8

```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You need an ESP32 and the wheelchair to run it.
```bash
  python --version
  python -m venv nama_venv
  nama_venv\Scripts\activate
  pip install mediapipe
  pip install ultralytics
  pip install opencv-python
```

You will need 2 camera for this setup, if you confuse about how to change it, here example of changing the camera
```bash
cap_lstm = cv.VideoCapture(0)  # set the value to 0/1/2/3 depent on how many camera you have
cap_yolo = cv.VideoCapture(1)
```
## 🍿 Features

- Safe breaking system.
- Easy gesture sibi language for controlling wheelchair.



## Authors

- <img alt="Static Badge" src="https://img.shields.io/badge/AgungHari-black?style=social&logo=github&link=https%3A%2F%2Fgithub.com%2FAgungHari">
- <img alt="Static Badge" src="https://img.shields.io/badge/Deva-black?style=social&logo=github">


## License

<img alt="GitHub License" src="https://img.shields.io/github/license/AgungHari/Smart-Wheelchair-Control-System-Based-on-SIBI-Gesture-and-Smart-Braking-System-using-YOLOv8-and-LSTM">




# Introduction to Elephant Detection System

This Elephant Detection System to prevent human-elephant conflict and tracking of elephants using deep learning. The major aim of the study is to minimize the human-elephant conflict in the  rural areas of Sri Lanka to protect human lives from elephant attacks.

Sri Lanka is very famous for wild animals and its nature of beauty. Elephant is a key component of Sri Lankan tourism industry. However, elephants are dangerous in some circumstances. Human-wild elephant conflicts are the most frequently reported issue from various parts of our country. There have been many incidents in the past where conflict between humans and elephants has caused serious damage and resulted in the loss of lives as well as property.

Therefore, this research tries to innovate a prototype to prevent elephant attacks using deep learning and machine learning approaches. In this research, computer vision plays the major role. Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.

Basically, this research is trying to build a prototype (An Elephant Detection System) that identifies an elephant in the video frame. For an example, the video source might be coming from CCTV or other cameras, then the video will be analyzed frame by frame to identify elephants in it. A warning message is sent as soon as the position of the elephant overlaps with the area of conflict. Afterwards, the warning alarm will be turned on.

![Demo1](github-readme-content/diagram.jpg)



### Programming Languages & Frameworks
- Python
- SMTPLIB
- Playsound
- TensorFlow
- Protobuf
- TensorFlow Object Detection Framework

### Technical Requirements

- Python 3.8
- TensorFlow 2.5

## Configuration & Setup

This project supports CPU and GPU environments to run. If your computer has a GPU (Graphics Processing Unit) with CUDA-Enabled, You can setup a GPU-TensorFlow version to run this application much faster.

If you don't have a GPU, you can still install this application with CPU-TensorFlow version. However, the processing speed with CPU is much slower than the processing on GPU.

If your computer has a GPU, you need to install CUDA & CUDDN to enable the GPU support for TensorFlow. Installation process of CUDA & CUDDN is a bit tricker than the normal TensorFlow installation.

### 01. Installing CUDA & CUDDN

If your computer has no GPU, You should skip this installation, you can run this application with CPU. However, It will be much slower. If your computer has a GPU but, you don't like to install CUDA & CUDDN, you can still skip this process. But, this application will only use CPU for tensor processing even though you have a hardware-based GPU. Therefore, CUDA & CUDDN installation process is very essential to enable the GPU support for computer that a GPU.

Note: If your computer has no GPU, You shouldn't install CUDA & CUDDN. You can skip this installation and go ahead with next step.


#### Checking suitable CUDA & CUDDN verions for TensorFlow.

Check Here:- [https://www.tensorflow.org/install/source#gpu](https://www.tensorflow.org/install/source#gpu)

![TensorFlow Versions](github-readme-content/tensorflow-version.jpg)

We are going to install TensorFlow (V 2.7.0), Therefore, we need to install CUDA (V 8.1) and CUDDN (V 11.2) according to the chart above.

First, we need to install CUDA (V 8.1) then, we should install CUDDN (V 11.2) for the TensorFlow (V 2.7.0).

The above chart clearly explains which CUDA & CUDDN versions that we should install for the selected TensorFlow.



#### Installing CUDA

Download CUDA From Here:- [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

![TensorFlow Versions](github-readme-content/cuda-version.jpg)

Download & Install the appropriate CUDA version from the official website of NVIDIA.The installation process is simple, we can install it with the GUI installer.

![TensorFlow Versions](github-readme-content/cuda-install.png)


#### Installing CUDDN
Download CUDDN From Here:- [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)

![TensorFlow Versions](github-readme-content/cuddn-install.jpg)

Download & Install the appropriate CUDDN version from the official website of NVIDIA.This time, the installation process is bit tricky.

01.Download and Extract CUDDN zip.

02.There are 3 following folders in it.
 * bin
 * include
 * lib

03.Open the CUDA Installation directory (WHERE is the CUDA installed) Ex: (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2)

![TensorFlow Versions](github-readme-content/cuda-install2.png)

04.Copy all content from bin, include, lib folder from the extracted CUDDN zip to the installed CUDA directory's bin, include, lib. An example is shown above.

#### Add CUDA & CUDDN Path to the environment variable.

After that installation of CUDA & CUDDN, we need to add the path to the environment variable. Go to PC settings and add them. If you don't know how to add them, just google it.

01.We need to add the following path.

 * Bin Path
 * Libnvp Path

 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin

 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\libnvvp\


Note: Make sure to add the correct path.



![TensorFlow Versions](github-readme-content/env-path.jpg)

#### Check CUDA & CUDDN installed successfully.

- Open CMD and Type

  ```
    nvidia-smi
  ```
![TensorFlow Versions](github-readme-content/check-cuda-install.jpg)


If it shows like above, that means you have successfully installed CUDA in your computer.

### 02. Installing TensorFlow

Now, It's time to install TensorFlow. It is simple now. Just open CMD and type following command, that will install TensorFlow for both environments including GPU.

Note: If your computer does not have a GPU, You can still install it. It will only install the CPU version, if it does not find CUDA & CUDDN.

- Installing TensorFlow

  ```
    pip install tensorflow
  ```

To check whether this installed TensorFlow utilize the GPU. Open python interpreter and write following code. It will show the GPU memory.  


- Check TensorFlow uses GPU
  ```
  import tensorflow as tf
  print(tf.test.gpu_device_name())
  ```


Note: If your computer does not have a GPU, you need to check it. You can simply skip everything, including CUDA & CUDDN installation and only install "pip install tensorflow" and go ahead with the next step.

### 03. Installing Object Detection Framework

In order to install the object detection framework, clone the following repository. This repo contains Google Research of TensorFlow and Object Detection essential Contents.

Clone this Google models repository:- [https://github.com/tensorflow/models](https://github.com/tensorflow/models)

This repo is large (700MB). it contains almost all files of the TensorFlow and Google Object Detection Framework researches, we should install it manually to utilize the object detection framework.

- Clone models repo

  ```
    git clone https://github.com/tensorflow/models.git
  ```


Now, we have to download Protobuf to convert the serialize structured data.


Download Protobuf:- [https://github.com/protocolbuffers/protobuf/releases](https://github.com/protocolbuffers/protobuf/releases)

01.Extract it.

02.Copy the protoc.exe and paste it under the cloned repo (models/research) folder.


- From models/research folder, Execute the following code in CMD

  ```
    protoc object_detection/protos/*.proto --python_out=.
  ```

03.To install the object detection manually, copy the setup.py that is under (models/research/object_detection/packages/tf2/setup.py) and paste in the (models/research) folder.


- Install object detection manually from (models/research).

  ```
    pip install -e .
  ```

04.After the installation, we don't need the cloned models repo anymore.

### 04. Installing Other Frameworks


- Installing TF-Slim

  ```
    pip install tf-slim
  ```

- Installing tf-models-official

  ```
    pip install tf-models-official
  ```

- Installing Mediapipe

  ```
    pip install mediapipe
  ```

- Installing Opencv

  ```
    pip install opencv-python
  ```

- Installing Playsound

  ```
    pip install playsound
  ```


Note: If you encounter any module missing issue, just try with "pip install MODULENAME", if it does not work try search on Google for answers.

### Running the system (Execution)

Now, it's time to run our app. We have installed all of the important frameworks. If you find any issues on installation, it's better to do a Google search.

- Clone the repo

```
  git clone https://github.com/GunarakulanGunaretnam/elephant-detection-system-tensorflow.git

```
 This repo contains 4 different projects related to elephant detection system.

 ### Project Directory layout

    .
    ├── 0-elephant-detection-only
    ├── 1-elephant-detection-with-alarm-and-email-alert
    ├── 2-elephant-detection-with-tensorflow-lite
    ├── 3-elephant-detection-custom-trained-model


 #### 0-elephant-detection-only
 =====================

This folder contains the basic elephant detection system. It does only identifying the elephant in video frame. It has no alarming or email sending features. It only has detection part.

This elephant detection is using the ssd_mobilenet_v2_320x320_coco17_tpu-8 dataset model that was trained with COCO-2017 dataset.

This model has more than 80 objects in it. We programmatically muted other objects and allow this model to predict elephant only.

 ##### Running

 ```
  python run.py

 ```

 ##### To change video source or webcam.

![TensorFlow Versions](github-readme-content/0-elephant-detection-only-webcam.jpg)

Note:- We can give the path of the video to process. If we want to use it as a real time webcam. We can simply put the index of the camera

Example:-

```
 cap = cv2.VideoCapture(0)

```

Note:- 0 is the first camera, 1 is the second camera and 2,3 and etc.


#### 1-elephant-detection-with-alarm-and-email-alert
=======================================

This folder contains the complete elephant detection system with alarming and email sending feature. It is also using the same approach like 0-elephant-detection-only project. But, It has alarming and email sending features.

##### Running

```
 python run.py

```

##### Configuration of email

![TensorFlow Versions](github-readme-content/1-elephant-detection-with-alarm-and-email-alert-email.jpg)

Note:- Here, we have to give the sender-email, sender-password and receiver-email. The sender email is the email that the system is using, the receiver email is the email where the warning message are sent to.


![TensorFlow Versions](github-readme-content/less.jpg)

Note:- We have to turn on the secure less option  on to in sender Gmail. Otherwise, we are unable to send emails.


##### To change video source or webcam.

![TensorFlow Versions](github-readme-content/1-elephant-detection-with-alarm-and-email-alert-webcam.jpg)

Note:- We can give the path of the video to process. If we want to use it as a real time webcam. We can simply put the index of the camera

Example:-

```
 cap = cv2.VideoCapture(0)

```

#### 2-elephant-detection-with-tensorflow-lite
=================================

This project is exactly same as the "1-elephant-detection-with-alarm-and-email-alert", But, It is using the tflite model instead of ssd_mobilenet_v2_320x320_coco17_tpu.

TensorFlow Lite (TFLITE) is enabling the deep learning ability in smaller computing devices such as mobile, embedded, and IoT devices.

```
 python run.py

```
* [More info about TFLITE](https://www.tensorflow.org/lite/guide/python)


#### 3-elephant-detection-custom-trained-model
====================================

This project is using a custom trained model. The previous projects are using the ssd_mobilenet_v2_320x320_coco17_tpu COCO-2017 dataset model that has more than 80 objects in it. But, this model customized with only one object that is elephant. This moldel fasther than other models since it has only on object.

```
 python run.py

```


### Output


![Demo1](github-readme-content/demo1.jpg)

![Demo2](github-readme-content/demo2.jpg)



Social Media Links
---

* [Linkedin Profile](https://www.linkedin.com/in/gunarakulangunaretnam/)
* [Facebook Page](https://www.facebook.com/gunarakulangunaretnam)
* [Twitter Profile](https://twitter.com/gunarakulang)
* [Instagram Profile](https://www.instagram.com/gunarakulangunaretnam/)
* [Youtube Channel](https://www.youtube.com/channel/UCMWkED5sabgVZSCKjZuRJXA)


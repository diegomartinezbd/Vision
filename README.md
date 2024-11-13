
# Vision

## Overview
This document provides an overview of different computer vision techniques used for image processing, including their objectives, neural networks used, and real-world applications.

## Summary of how these techniques are applied in image processing:

| **Task**               | **Objective**                               | **Networks Used**               | **Use Cases**                                    |
|------------------------|---------------------------------------------|---------------------------------|--------------------------------------------------|
| **Image Classification**| Assign a label to the entire image          | Traditional CNNs (LeNet, AlexNet) | Medical diagnosis, object classification in photos |
| **Object Detection**    | Locate objects and classify them            | R-CNN, YOLO, Faster R-CNN        | Autonomous vehicles, security cameras            |
| **Image Segmentation**  | Assign labels to each pixel                 | FCN, U-Net                       | Medical images, agriculture, autonomous vehicles |
| **Image Generation**    | Create realistic or transformed images      | GANs, VAEs                       | Generative art, deepfakes, text-to-image generation |

## Examples of Object Detection
Here are a couple of examples of object detection applications:
* [Hand Detection](https://gerardomunoz.github.io/Vision/Hand_Detection.html) - A demo of detecting hands in real-time.
* [Hand Detection Landmark](https://gerardomunoz.github.io/Vision/Hand_Detection_LandMark.html) - Example showcasing hand landmark detection for gesture recognition.

## Challenge
Check out this video on building an object detection: [Object Detection Challenge](https://www.youtube.com/watch?v=sGDJwilP-oo).

## Image Classification

| Feature                 | **LeNet**                           | **AlexNet**                                  |
|-------------------------|-------------------------------------|----------------------------------------------|
| **Year**                | 1989                                | 2012                                         |
| **Developed By**        | Yann LeCun and team                 | Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton |
| **Input Image Size**    | 32x32 grayscale                     | 224x224 RGB                                  |
| **Number of Parameters**| ~60,000                             | ~60 million                                  |
| **Layers**              | 2 Conv, 2 Pooling, 2 FC             | 5 Conv, 3 Pooling, 3 FC                      |
| **Activation Function** | tanh                                | ReLU                                         |
| **Pooling**             | Average Pooling                     | Max Pooling                                  |
| **Regularization**      | None                                | Dropout                                      |
| **Training**            | CPU                                 | GPU                                          |
| **Dataset**             | MNIST (Handwritten Digits)          | ImageNet (Real-world objects, 1000 classes)  |
| **Legacy**              | First practical CNN                 | Popularized deep learning, sparked modern AI research |

https://www.kaggle.com/code/gerardomunoz/seminario-2019-x/notebook

## Object Detection
https://github.com/GerardoMunoz/Vision/blob/main/GroundingDINO.ipynb


## Image Generation
https://github.com/GerardoMunoz/LinearCombinationGrade/blob/master/CalifCombLin_2020_4_09_wea001.ipynb

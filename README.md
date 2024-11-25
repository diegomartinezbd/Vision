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
* https://github.com/GerardoMunoz/Vision/blob/main/GroundingDINO.ipynb
* https://github.com/GerardoMunoz/Vision/blob/main/Zero_shot_object_detection_with_grounding_dino.ipynb

## Image Segmentation
* https://github.com/GerardoMunoz/Vision/blob/main/Automatic_mask_generator_example.ipynb

## Image Generation
* https://github.com/GerardoMunoz/LinearCombinationGrade/blob/master/CalifCombLin_2020_4_09_wea001.ipynb

## Comparison of Lightweight TensorFlow.js Models for Efficient Image Processing
| **Model Name**         | **Image Suitability**                    | **Speed for Prediction**        | **Speed for Retraining Last Layers** | **Why Choose It**                                                                                          |
|-------------------------|------------------------------------------|----------------------------------|---------------------------------------|------------------------------------------------------------------------------------------------------------|
| **COCO-SSD**           | Best for object detection (e.g., people, objects) | Fast (real-time on most devices) | Moderate                              | Lightweight and optimized for speed.                                                                      |
| **MobileNet**          | Best for image classification            | Very Fast                        | Fast                                  | Pre-trained on ImageNet; great for classification tasks, lightweight, and fast inference.                 |
| **YOLO-Tiny**          | Best for simple object detection         | Fast (real-time on mobile)       | Moderate                              | Optimized for smaller devices; good balance between speed and object detection accuracy.                   |
| **PoseNet** (Optional) | Best for pose estimation and keypoints   | Moderate                         | Slow                                  | Useful for pose estimation; lightweight and well-suited for quick applications like tracking skeletons.    |


# **Acne Severity Detection Project Overview**

## **Environment**

- **Python Version:** 3.6
- **Required Modules:**
    - cmake==3.26.3
    - cntk==2.7
    - dlib==19.24.1
    - imageio==2.15.0
    - joblib==1.1.1
    - numpy==1.19.5
    - opencv-python==4.6.0.66
    - pandas==1.1.5
    - Pillow==8.4.0
    - scikit-build==0.16.7
    - scikit-learn==0.24.2
    - scipy==1.5.4

## **Required Data Files & Model**

- **Cascade Models:**
    - haarcascade_eye.xml
    - shape_predictor_68_face_landmarks.dat
- **Pretrained Model:**
    - ResNet152_ImageNet_Cafee.model

## **Project Steps**

1. **Acne Severity Classification:**
    - Classify acne severity levels ranging from 1 to 5.
2. **Image Processing:**
    - Utilize face-landmark and eye-cascade models to crop images into specific facial regions.
    - Handle cases where faces are not detected or multiple faces are present.
    - Distinguish between frontal and side-facing views.
3. **Expert Labeling:**
    - Expert dermatologists assign acne severity scores (1 to 5) to cropped images based on facial regions.
4. **Data Augmentation:**
    - Increase training data by rolling images, altering x or y coordinates.
5. **Image Resizing:**
    - Resize images to 224 x 224 to match the input size required for the ResNet-152 model.
6. **Data Splitting:**
    - Split images into 80% for training and 20% for validation to train the neural network.
7. **Random Seed Initialization:**
    - Set random seed to 5 for consistent weight initialization and data split randomness.
8. **Neural Network Model Building:**
    - Use CNTK's pretrained ResNet-152 model and labeled images to build a neural network for acne classification.
9. **New Image Processing:**
    - Crop new images into facial regions.
10. **Scoring with Models:**
    - Utilize ResNet152_ImageNet_Caffe.model and the acne classification model to obtain severity scores for each facial region.
11. **Result Export:**
    - Export the obtained scores to Excel for further analysis and reporting.

By following these steps, the project aims to create a robust system for acne severity detection, combining image processing, data augmentation, and neural network modeling. The use of pretrained models and expert labeling ensures a comprehensive and accurate classification of acne severity levels.

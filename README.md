# 🌱 Plant Disease Detection with Convolutional Neural Networks (CNN)

## Overview

This project uses a Convolutional Neural Network (CNN) to detect and classify plant diseases based on images of plant leaves. The model is trained on a labeled dataset and achieves high accuracy in identifying diseases such as Potato Early Blight, Tomato Bacterial Spot, and Corn Common Rust.

By integrating advanced deep learning techniques, this project aims to provide a reliable tool to assist farmers and agricultural professionals in diagnosing plant diseases efficiently, potentially increasing crop yield and minimizing losses.

## Features

🖼️ Image-based Classification: Predict plant diseases using leaf images.
🔍 Accurate Predictions: Achieves over 99% accuracy on test data.
🧠 Deep Learning Model: Built using TensorFlow and Keras frameworks.
📊 Visualization Tools: Includes accuracy/loss plots and prediction comparison.
🌐 Streamlit App: A user-friendly interface to classify plant diseases with ease.

## Dataset

The dataset used in this project contains images of plant leaves categorized into three classes:

Potato Early Blight
Tomato Bacterial Spot
Corn Common Rust
The dataset includes 900 images (300 per class). Images are resized to 256x256 pixels and normalized for model training.

## Project structure

├── data/                    # Folder for dataset images
├── models/                  # Saved models and weights
├── app/                     # Streamlit application files
├── notebooks/               # Jupyter notebooks for model training and analysis
├── requirements.txt         # Python dependencies
├── plant_disease.h5         # Trained model in H5 format
├── README.md                # Project documentation
└── plant_model.json         # Model architecture in JSON format


## Installation

Prerequisites
Ensure you have Python 3.8+ installed. Then, clone the repository:

git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection

Install Dependencies
Install all required packages using the requirements.txt file:

pip install -r requirements.txt


## Usage

Training the Model
Prepare the dataset and place it in the data/ directory.
Run the Jupyter notebook notebooks/train_model.ipynb to train the CNN.
Testing the Model
To evaluate the model, use the test data in the same notebook or run:

python evaluate_model.py

Running the Streamlit App
Launch the app for real-time predictions:

streamlit run app/main_app.py
Upload an image of a plant leaf, and the app will predict the disease.

## Results

Training Accuracy: ~99.85%
Validation Accuracy: ~99%
Test Accuracy: ~99.44%
Model Performance Visualization
Training vs. Validation Accuracy Plot
Training vs. Validation Loss Plot

## Future Enhancements

🌍 Expand the dataset to include more plant species and diseases.
📱 Deploy the model as a mobile application for wider accessibility.
⚙️ Add more advanced models like ResNet or EfficientNet for improved accuracy.




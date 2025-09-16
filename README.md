# CodeAlpha_Handwritten-Character-Recognition


This project implements a Convolutional Neural Network (CNN) model for recognizing handwritten digits using the MNIST dataset. The goal is to build a model that can accurately classify images of handwritten digits from 0 to 9.

## Project Overview

The project follows these steps:

1.  **Data Loading and Exploration**: Load the MNIST dataset and visualize sample images and their corresponding labels.
2.  **Data Preprocessing**: Prepare the data for the CNN model by normalizing pixel values, reshaping images, and one-hot encoding labels.
3.  **Model Building**: Define a CNN architecture using `tensorflow.keras`, including convolutional, pooling, and dense layers.
4.  **Model Compilation and Training**: Compile the CNN model with an appropriate loss function and optimizer, and train it on the preprocessed training data.
5.  **Model Evaluation**: Evaluate the trained model's performance on the test dataset to assess its accuracy and loss.
6.  **Prediction and Visualization**: Use the trained model to make predictions on new handwritten digit images and visualize the results, comparing true and predicted labels.

## Dataset

The project uses the **MNIST dataset**, a widely used dataset for handwritten digit recognition. It consists of 60,000 training images and 10,000 testing images of handwritten digits.

## Model Architecture

The CNN model architecture includes:

*   Convolutional layers with ReLU activation for feature extraction.
*   MaxPooling layers for down-sampling and reducing spatial dimensions.
*   A Flatten layer to convert the 2D feature maps into a 1D vector.
*   Dense layers with ReLU activation for classification.
*   An output Dense layer with Softmax activation for predicting the probability distribution over the 10 digit classes.

## Results

The trained model achieved a test accuracy of approximately **[99.19%]**. The training history and evaluation results are included in the notebook.

## How to Run

1.  Clone this repository.
2.  Open the Jupyter notebook (or Colab notebook).
3.  Run the cells sequentially to execute the code for data loading, preprocessing, model building, training, evaluation, and prediction.

## Dependencies

The project requires the following libraries:

*   TensorFlow
*   Keras
*   Matplotlib
*   NumPy

These dependencies can be installed using pip:

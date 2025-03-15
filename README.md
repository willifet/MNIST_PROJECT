# MNIST Handwritten Digit Classification

### Overview

The MNIST (Modified National Institute of Standards and Technology) dataset is a collection of 70,000 grayscale images of handwritten digits (0-9). It is widely used as a benchmark dataset for image classification and deep learning models.

This project aims to build a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify the handwritten digits in the MNIST dataset with high accuracy.
### Dataset Details

Total Samples: 70,000 images

Training Set: 60,000 images

Test Set: 10,000 images

Image Size: 28x28 pixels

Number of Classes: 10 (digits 0-9)

Format: Grayscale images
### Installation
Ensure you have Python 3.7+ installed, along with the following dependencies:

pip install tensorflow, numpy matplotlib

### Model Architecture
The model is a Convolutional Neural Network (CNN) consisting of:

Conv2D layers to extract features

MaxPooling layers to reduce spatial dimensions

Fully connected layers to classify the digits

Softmax activation for multi-class classification

### Training the Model
Run the following command to train the model:
python src/train.py

### Evaluating the Model
To test the trained model on the test dataset:
python src/test.py

### Making Predictions
To make prediction on new images:
python src/predict.py --image path/to/image.png

### Results 
After training for 5 epochs, the model achieves an accuracy of approximately 96% on the test dataset.

### Future Improvements

Implement data augmentation to improve generalization.

Try different architectures like ResNet, VGG, or Transformer models.

Optimize hyperparameters (learning rate, batch size, number of layers).

Deploy the model using a Flask or FastAPI web application.

### References


Original MNIST Dataset

TensorFlow Documentation

### License
This project is open-source and available under the MIT License.

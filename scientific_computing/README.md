# CNN for CIFAR-10 Image Classification

## 1. Overview

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using TensorFlow/Keras.

The model combines data augmentation, convolutional feature extraction, batch normalization, pooling layers, dropout regularization, and fully connected layers to classify images into 10 object categories.

Dataset: CIFAR-10

Classes:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

---

# 2. Model Architecture

The CNN architecture is organized as follows:

Input (32×32×3)
↓
RandomFlip
↓
RandomRotation
↓
RandomZoom
↓
Conv2D(32)
↓
BatchNormalization
↓
MaxPooling2D
↓
Conv2D(64)
↓
BatchNormalization
↓
MaxPooling2D
↓
Conv2D(128)
↓
BatchNormalization
↓
MaxPooling2D
↓
Flatten
↓
Dense(256, ReLU)
↓
Dropout(0.5)
↓
Dense(10, Softmax)

---

## Data Augmentation

To improve generalization and reduce overfitting, online data augmentation is applied during training:

* Random horizontal flip
* Random rotation (±10%)
* Random zoom (±10%)

These transformations generate different variations of training images at each epoch.

---

## Convolution Blocks

### Block 1

* Conv2D (32 filters, 3×3)
* Batch Normalization
* Max Pooling

Learns low-level features such as:

* edges
* corners
* textures

### Block 2

* Conv2D (64 filters, 3×3)
* Batch Normalization
* Max Pooling

Learns intermediate features such as:

* wheels
* eyes
* wings
* object parts

### Block 3

* Conv2D (128 filters, 3×3)
* Batch Normalization
* Max Pooling

Learns high-level semantic features such as:

* animal heads
* vehicle structures
* object shapes

---

## Classification Head

### Flatten

Converts feature maps into a one-dimensional feature vector.

### Dense Layer

Dense(256, ReLU)

Combines extracted visual features and learns class-specific representations.

### Dropout

Dropout rate: 0.5

Randomly disables 50% of neurons during training to reduce overfitting.

### Output Layer

Dense(10, Softmax)

Produces class probabilities for the 10 CIFAR-10 categories.

---

# 3. Data Preprocessing

## Dataset Loading

The CIFAR-10 dataset is loaded directly from Keras:

* 50,000 training images
* 10,000 testing images

## Normalization

Pixel values are scaled from:

0–255

to:

0–1

using:

x = x / 255.0

## Label Encoding

Labels are converted to one-hot vectors using:

to_categorical()

Example:

Class "cat"

becomes:

[0,0,0,1,0,0,0,0,0,0]

---

# 4. Training Configuration

## Hyperparameters

* Batch Size = 64
* Epochs = 20
* Learning Rate = 0.001
* Number of Classes = 10
* Image Size = 32×32

## Optimizer

Adam Optimizer

Learning Rate:

0.001

## Loss Function

Categorical Cross-Entropy

Used for multi-class classification.

## Evaluation Metric

Accuracy

---

# 5. Training Callbacks

## Early Stopping

Monitors:

validation accuracy

Configuration:

* patience = 5
* restore_best_weights = True

Stops training when validation accuracy does not improve for 5 consecutive epochs.

---

## Model Checkpoint

Automatically saves the best-performing model:

best_cnn_model.keras

based on validation accuracy.

---

## Reduce Learning Rate on Plateau

Monitors:

validation loss

Configuration:

* factor = 0.5
* patience = 3

Reduces learning rate when validation loss stops improving.

---

# 6. Evaluation

## Test Evaluation

The trained model is evaluated on the CIFAR-10 test set.

Metrics:

* Test Accuracy
* Test Loss

---

## Confusion Matrix

A normalized confusion matrix is generated to analyze classification performance across all classes.

Output file:

confusion_matrix.png

---

## Training Curves

Accuracy and loss curves are plotted for:

* Training Set
* Validation Set

Output file:

training_history.png

---

## Sample Predictions

The model visualizes predictions on sample test images.

Correct predictions are shown in green.

Incorrect predictions are shown in red.

Output file:

sample_predictions.png

---

# 7. Output Files

Generated files:

* best_cnn_model.keras
* confusion_matrix.png
* training_history.png
* sample_predictions.png

---

# 8. Workflow

Load CIFAR-10 Dataset
↓
Normalize Images
↓
One-Hot Encode Labels
↓
Apply Data Augmentation
↓
CNN Feature Extraction
↓
Dense Classification Layers
↓
Model Training
↓
Validation Monitoring
↓
Save Best Model
↓
Test Evaluation
↓
Confusion Matrix
↓
Training Visualization
↓
Sample Predictions

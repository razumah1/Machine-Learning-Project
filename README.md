# Machine-Learning-Project

## Overview
This repository contains the Jupyter Notebook `Richmond_Azumah_final_project_code.ipynb`, which implements semi-supervised learning techniques for classifying handwritten digits from the MNIST dataset. The notebook explores methods such as baseline supervised learning, entropy minimization, and pseudo-labeling to improve classification performance.

## Key Features
1. **Baseline Supervised Learning**: Trains a neural network using a small labeled dataset.
2. **Entropy Minimization**: Enhances the model by incorporating an entropy minimization loss function to leverage unlabeled data.
3. **Pseudo-Labeling**: Uses predictions on unlabeled data as pseudo-labels to augment the training set.
4. **Performance Metrics**: Evaluates model performance using accuracy, precision, recall, F1 score, and confusion matrix.
5. **Visualizations**: Includes loss plots and confusion matrices for insights into model performance.

## Prerequisites
To run the notebook, ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or Jupyter Lab
- Required Python libraries:
  - numpy
  - matplotlib
  - tensorflow
  - sklearn
  - seaborn

Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage Instructions
1. Clone this repository:
```bash
git clone [repository_url]
```
2. Navigate to the project directory:
```bash
cd [project_directory]
```
3. Launch Jupyter Notebook:
```bash
jupyter notebook
```
4. Open the `Richmond_Azumah_final_project_code.ipynb` file and execute the cells sequentially.

## Structure
The notebook is organized into the following sections:

1. **Setup and Imports**: Loads required libraries, including TensorFlow, PyTorch, and sklearn.
   ```python
   import numpy as np
   from array import array
   import matplotlib.pyplot as plt
   import tensorflow as tf
   from tensorflow.keras import layers, models
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import accuracy_score
   import torch.nn as nn
   import torch.optim as optim
   from torchvision import datasets, transforms
   ```

2. **Data Preparation**:
   - **MNIST Dataset**:
     - Loads the MNIST dataset.
     - Normalizes and reshapes images.
     - Splits data into labeled and unlabeled subsets for semi-supervised learning.
     ```python
     mnist = tf.keras.datasets.mnist
     (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
     train_images = train_images.reshape(-1, 784).astype('float32') / 255.0
     test_images = test_images.reshape(-1, 784).astype('float32') / 255.0
     ```
   - **Two Moons Dataset**:
     - Generates a synthetic two-class dataset using sklearn's `make_moons` function.
     - Applies preprocessing to split the data into training and testing subsets.
     ```python
     from sklearn.datasets import make_moons
     X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

3. **Baseline Supervised Learning**:
   - Implements a simple two-layer neural network.
   - Trains the model using only labeled data.
   ```python
   model = models.Sequential([
       layers.Dense(200, activation='relu', input_shape=(784,)),
       layers.Dense(10, activation='softmax')
   ])
   ```

4. **Entropy Minimization**:
   - Defines a custom loss function that incorporates entropy minimization.
   - Combines labeled and unlabeled data to train the model.
   ```python
   def entropy_minimization_loss(y_true, y_pred):
       entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-8), axis=-1)
       return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + lambda_value * entropy
   ```

5. **Pseudo-Labeling**:
   - Predicts labels for unlabeled data and combines them with labeled data.
   - Retrains the model with the augmented dataset.
   ```python
   pseudo_labels = model.predict(unlabeled_images)
   pseudo_labels = np.argmax(pseudo_labels, axis=1)
   ```

6. **Evaluation and Visualization**:
   - Evaluates the model on the test set using metrics like accuracy, precision, recall, and F1 score.
   - Visualizes results through confusion matrices and training loss plots.
   ```python
   plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
   ```

## Data
The project uses the following datasets:
- **MNIST Dataset**:
  - 60,000 training samples and 10,000 testing samples.
  - Images are grayscale and have a resolution of 28x28 pixels.
- **Two Moons Dataset**:
  - A synthetic dataset with two interleaving moon-shaped classes.
  - 1,000 samples with added noise for complexity.

## Contact
For questions or collaboration, reach out to:

**Richmond Ewenam Kwesi Azumah**  
Email: [insert email here]  
LinkedIn: [insert LinkedIn profile here]


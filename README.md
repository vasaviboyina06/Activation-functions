# Activation-functions
performance of  different activation functions on the dataset.

From Sigmoid to Swish: A Deep Dive into Activation Functions in Neural Networks

Overview

This study involves developing, training, and comparing neural networks using various activation functions (ReLU, Sigmoid, Tanh, Swish, Softmax, and LeakyReLU) on a bank churn prediction dataset. The dataset includes customer information and labels indicating if a client churned. The model predicts churn based on input characteristics.

Features
Implements a variety of activation functions for comparative performance evaluation.
Metrics for training, validation, and testing (accuracy and mean absolute error).
Creates representations of accuracy and loss throughout epochs.
Determines the optimal activation function based on test accuracy and MAE.

Technologies Used
Python Libraries:
TensorFlow/Keras: Building and training neural networks.
Pandas: Data handling and manipulation.
NumPy: Numerical computations.
Matplotlib: Plotting and visualization.
Scikit-learn: Data preprocessing and evaluation.
Machine Learning Techniques:
Standardization
One-hot encoding and label encoding
Neural network architecture with various activation functions

Dataset
The dataset used is a cleaned version of a bank churn prediction dataset with the following features:

Input Features: Customer attributes like Geography, Gender, Credit Score, Balance, etc.
Output Label: A binary variable indicating churn (1) or no churn (0).

How to Run the project

1. clone this repository
   git clone https://github.com/your-repo/bank-churn-activation-functions.git
cd bank-churn-activation-functions
2. Install required dependencies:
    pip install -r requirements.txt
3. Run the script
   python main.py
   
Visualizations
Accuracy and Loss Curves: Graphs show the training and validation accuracy/loss for each activation function.
Comparative Plots: The top performance is determined by plotting validation accuracy and loss across activation functions.

Results
Best Activation Function: ReLU achieved the highest test accuracy and generalization.
Metrics:
S.no	Activation function	Accuracy(%)	   MAE
1      	Relu 	             86.60	       0.2018
2	      Tanh	             87.40	       0.1995
3	      Sigmoid	           85.33	       0.2401
4	      Swish	             86.40	       0.2069
5	      Leakyrelu	         86.00	       0.2098
6	      SoftMax	           85.93	       0.2208

# aminpri-isthakur-rganesh-a4

Part 1: K-Nearest Neighbors Classification 

Problem Formulation:

The task at hand is to implement a k-nearest neighbors (k-NN) classifier from scratch. K-nearest neighbors is a non-parametric supervised machine learning algorithm used for both classification and regression tasks. In this case, the focus is on classification.
We must implement our own program and compare the accuracy scores with the scikit-learn implementation on two different datasets, iris and digits.
The accuracy scores generated for the implementation and scikit-learn's implementation will be compared. If correctly implemented, the accuracy scores will be very close for all tested cases.

Program:

The Python code defines a KNearestNeighbors class, implementing a K-Nearest Neighbors classifier from scratch. 
The class takes parameters such as the number of neighbors (n_neighbors), weight function ('uniform' or 'distance'), and distance metric ('l1' or 'l2'). 
The fit method trains the model with input data and true class labels. 
The predict method uses the fitted model to predict class labels for a given test dataset. It calculates distances between samples, considers neighbors based on the specified parameters, assigns weights, and predicts the class with the highest weighted occurrences among neighbors. The code utilizes Euclidean or Manhattan distance functions from utils.py


Part 2: Multilayer Perceptron Classification 

Problem Formulation:

The task is to implement a feedforward fully connected multilayer perceptron classifier from scratch with the following specifications:
Neural Network Architecture:
Three layers: Input layer, Hidden layer, and Output layer.
Feedforward and fully-connected.
Number of neurons in the hidden layer specified by the parameter n_hidden.
Activation functions for hidden layer can be one of: 'identity', 'sigmoid', 'tanh', or 'relu'.

Training Process:
Training occurs using batch gradient descent.
During each iteration, forward propagation is performed to produce an output by the output layer.
The cross-entropy loss is calculated using the output and stored in a history list.
Output is compared to the expected output (target class values), and an error is calculated.
Backward propagation is performed, and weights are updated based on the contribution to the error.
Learning rate is used to control the extent of model changes in response to the estimated error.

Data Preparation:
Target class values are categorical, and neural networks require numerical data.
Categorical data is converted to numerical representation using one-hot encoding.
One-hot encoding creates an array where each column represents a possible categorical value, and each row has 0s or 1s indicating the presence of a specific class value.

Testing:
The provided driver program (main.py) is run to test the implementation.
The accuracy scores of the implementation are compared with the scikit-learn implementation on the datasets (Iris and Digits).

Challenges:
Getting all the codes to work together seamlessly posed a challenge since each code relied on the others. If one had an error, it would impact the entire system. Additionally, optimizing the code for efficiency to reduce runtime presented its own set of challenges.

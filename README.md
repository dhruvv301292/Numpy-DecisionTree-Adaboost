# NumpyAdaboost
A decision stump weak classifier based Adaboost implementation for binary classification

Implemented the adaboost algorithm from scratch

The dataset for this task is synthetically generated and has two entries xi = (x1, x2) and y. Here xi =
(x1, x2) ∈ [−2, 2] × [−2, 2] is the i-th instance and yi ∈ {−1, 1} is its corresponding label. The training and
test dataset are in ”train-adaboost.csv” and ”test-adaboost.csv” respectively

The following functions are used:
(a) read data: This function reads the data from the input file.
(b) weak classifier: This function finds the best weak classifier, which is a 1-level decision tree.
Defining n as number of training points and d as the number of features per data point, the
inputs to the function should be the input data (n × d array), the true labels (n × 1 array) and
the current distribution D (n × 1 array), it returns the best weak classifier with the
best split based on the error. Some of the things included in the output are: best feature
index,best split value, label, value of βt and predicted labels by the best weak classifier.
(c) update weights: This function computes the updated distribution Dt+1, The inputs to this
function are the current distribution D (n × 1 array), the value of βt, the true target values
(n × 1 array) and the predicted target values (n × 1 array). The function outputs the
updated distribution D.
(d) adaboost predict: This function returns the predicted labels for each weak classifier. The
inputs: the input data (n × d array), the array of weak classifiers (T × 3 array) and the array of
βt for t = 1 . . . T (T × 1 array) and output the predicted labels (n × 1 array)
(e) eval model: This function evaluates the model with test data by measuring the accuracy. Assuming we have m test points, the inputs are the test data (m × d array), the true labels for test data (m × 1 array), the array of weak classifiers (T × 3 array), and the array of βt for
t = 1 . . . T (T × 1 array). The function outputs: the predicted labels (m × 1 array) and the
accuracy.
(f) adaboost train: This function trains the model by using AdaBoost algorithm.
Inputs:
- The number of iterations (T)
- The input data for training data (n × d array)
- The true labels for training data (n × 1 array)
- The input features for test data (m × 2 array)
- The true labels for test data (m × 1 array)
Output:
- The array of weak classifiers (T × 3 array)
- The array of βt for t = 1 . . . T (T × 1 array)

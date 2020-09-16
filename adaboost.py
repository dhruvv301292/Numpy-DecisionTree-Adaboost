import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    X_data = np.genfromtxt ( filename, delimiter=',', skip_header=1, usecols=(0, 1))
    y_data = np.genfromtxt ( filename, delimiter=',', skip_header=1, usecols=(2))
    return X_data, y_data


def majority(data):
    plus = data[data[:, 2] == 1]
    minus = data[data[:, 2] == -1]
    return 1 if len(plus) > len(minus) else -1


def weak_classifier(X, y, D):
    X_sorted = np.hstack((X, y[:, np.newaxis]))
    X_sorted = np.hstack((X_sorted, D[:, np.newaxis]))
    min_error = 10000
    for j in range(len(X[0])):
        for i in range(len(X_sorted)):
            left = X_sorted[X_sorted[:, j] <= X_sorted[i][j]]
            right = X_sorted[X_sorted[:, j] > X_sorted[i][j]]
            lefty = majority(left)
            righty = majority(right)
            error = 0
            for k in range(len(right)):
                error += right[k][3] if right[k][2] != righty else 0
            for k in range(len(left)):
                error += left[k][3] if left[k][2] != lefty else 0
            if error < min_error:
                min_error = error
                best_split_index = i
                best_split_value = X_sorted[i][j]
                best_split_feature = j
                maj_left = lefty
                maj_right = righty
    beta_t = 0.5 * np.log((1 - min_error)/min_error)
    return beta_t, best_split_feature, best_split_value, maj_left, maj_right, best_split_index, min_error


def update_weights(X, y, D, model_t):
    beta_t = model_t[0]
    split_feat = model_t[1]
    split_value = model_t[2]
    majleft = model_t[3]
    majright = model_t[4]
    y_h = np.asarray([0]*len(X))[:, np.newaxis]
    D_plus = np.asarray([0]*len(D))[:, np.newaxis]
    Z_norm = 0
    X_sorted = np.hstack((np.hstack((X, y[:, np.newaxis])), y_h))
    X_sorted = np.hstack((np.hstack((X_sorted, D[:, np.newaxis])), D_plus))
    for i in range(len(X_sorted)):
        if X_sorted[i][split_feat] <= split_value:
            X_sorted[i][3] = -1 if X_sorted[i][2] != majleft else 1
        if X_sorted[i][split_feat] > split_value:
            X_sorted[i][3] = -1 if X_sorted[i][2] != majright else 1
    for i in range ( len ( X_sorted ) ):
        X_sorted[i][5] = (X_sorted[i][4] * np.exp(-beta_t)) if X_sorted[i][3] == -1 else (X_sorted[i][4] * np.exp (beta_t))
        Z_norm += (X_sorted[i][4] * np.exp(-beta_t)) if X_sorted[i][3] == -1 else (X_sorted[i][4] * np.exp (beta_t))
    return np.asarray ( X_sorted[:, 5] / Z_norm )


def weak_predict(X_test, model):
    y_pred = [0] * len(X_test)
    beta_t = model[0]
    split_feature = int(model[1])
    split_value = model[2]
    maj_left = model[3]
    maj_right = model[4]
    for i in range(len(X_test)):
        y_pred[i] = maj_left if X_test[i][split_feature] <= split_value else maj_right
    return beta_t * np.asarray(y_pred)


def adaboost_train(num_iter, X_train, y_train):
    hlist = []
    D = np.asarray([1/len(X_train)] * len(X_train))
    for i in range(num_iter):
        wk_classifier = weak_classifier(X_train, y_train, D)
        print(i+1, wk_classifier)
        D = update_weights(X_train, y_train, D, wk_classifier)
        hlist.append(wk_classifier)
    return hlist


def eval_model(X_test, y_test, hlist):
    error = 0
    print("number of models: ", len(hlist))
    y_pred = np.asarray([0.0]*len(X_test))
    for i in range(len(hlist)):
        y_pred += weak_predict(X_test, hlist[i])
    for i in range(len(y_pred)):
        y_pred[i] = 1 if y_pred[i] >= 0 else -1
    for i in range(len(y_pred)):
        error += 1 if y_pred[i] != y_test[i] else 0
    accuracy = 1 - (error/len(y_pred))
    print("accuracy = ", accuracy)
    return accuracy



X_train, y_train = read_data("train_adaboost.csv")
X_test, y_test = read_data("test_adaboost.csv")
D = np.asarray([1/len(X_train)] * len(X_train))
# model = weak_classifier(X_train, y_train, D)
# print(model)
# print(model[3])
# print(model[4])

h_list = np.asarray(adaboost_train(100, X_train, y_train))
accuracy = []
for i in range(len(h_list)):
    accuracy.append(eval_model(X_test, y_test, h_list[:i+1]))
iters = np.linspace(1, 100, 100)
plt.plot(iters, accuracy)
plt.title("Accuracy vs number of weak classifiers")
plt.xlabel("Number of classifiers")
plt.ylabel("Accuracy")
plt.show()








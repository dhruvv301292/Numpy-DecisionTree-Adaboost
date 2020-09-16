import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    X_data = np.genfromtxt ( filename, delimiter=',', skip_header=1, usecols=(0, 1))
    y_data = np.genfromtxt ( filename, delimiter=',', skip_header=1, usecols=(2))
    return X_data, y_data


def weak_classifier(X, y, D):
    X_sorted = np.hstack((X, y[:, np.newaxis]))
    X_unsorted = np.hstack((X_sorted, D[:, np.newaxis]))
    min_error = 10000
    for j in range(len(X[0])):
        X_sorted = np.asarray(sorted(X_unsorted, key=lambda a: a[j]))
        for i in range(len(X_sorted)):
            left = X_sorted[X_sorted[:, j] <= X_sorted[i][j]]
            right = X_sorted[X_sorted[:, j] > X_sorted[i][j]]
            for l in range(2):
                error = 0
                errorcount = 0
                maj_left = -1 if l == 0 else 1
                maj_right = -1 * maj_left
                for k in range(len(right)):
                    error += right[k][3] if right[k][2] != maj_right else 0
                    errorcount +=1 if right[k][2] != maj_right else 0
                for k in range(len(left)):
                    error += left[k][3] if left[k][2] != maj_left else 0
                    errorcount += 1 if left[k][2] != maj_left else 0
                if error < min_error:
                    min_error = error
                    best_split_value = X_sorted[i][j]
                    best_split_feature = j
                    left_class = maj_left
                    right_class = maj_right
    beta_t = 0.5 * np.log((1 - min_error)/min_error)
    return beta_t, best_split_feature, best_split_value, left_class, right_class


def update_weights(X, y, D, model_t):
    beta_t = model_t[0]
    split_feat = model_t[1]
    split_value = model_t[2]
    majright = model_t[4]
    majleft = model_t[3]
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
    for i in range(len(X_sorted)):
        X_sorted[i][5] = (X_sorted[i][4] * np.exp(beta_t)) if X_sorted[i][3] == -1 else (X_sorted[i][4] * np.exp(-beta_t))
        Z_norm += (X_sorted[i][4] * np.exp(beta_t)) if X_sorted[i][3] == -1 else (X_sorted[i][4] * np.exp(-beta_t))
    return np.asarray(X_sorted[:, 5]/Z_norm)


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
        D = update_weights(X_train, y_train, D, wk_classifier)
        hlist.append(wk_classifier)
    return hlist


def eval_model(X_test, y_test, hlist):
    error = 0
    y_pred = np.asarray([0.0]*len(X_test))
    for i in range(len(hlist)):
        y_pred += weak_predict(X_test, hlist[i])
    for i in range(len(y_pred)):
        error += 1 if np.sign(y_pred[i]) != y_test[i] else 0
    accuracy = 1 - (error/len(y_pred))
    return accuracy


def main():
    X_train, y_train = read_data("train_adaboost.csv")
    X_test, y_test = read_data("test_adaboost.csv")
    h_list = np.asarray(adaboost_train(400, X_train, y_train))
    accuracytest = []
    for i in range(len(h_list)):
        accuracytest.append(eval_model(X_test, y_test, h_list[:i+1]))
    iters = np.linspace(1, 400, 400)
    plt.plot(iters, accuracytest)
    plt.title("Test Accuracy vs number of weak classifiers")
    plt.xlabel("Number of classifiers")
    plt.ylabel("Accuracy")
    plt.show()
    print("accuracy after training 400 weak classifiers: ", accuracytest[-1] * 100, "%")
    # accuracy after training 400 weak classifiers:  97.0 %

main()








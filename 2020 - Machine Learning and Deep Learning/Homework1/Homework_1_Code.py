# Politecnico di Torino
# 01TXFSM - Machine learning and Deep learning
# Homework 1
# Alberto Maria Falletta - s277971

# IMPORTS -----------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.svm import SVC

# Custom colormap for decision boundaries representation (Order: Green, Orange, Blue)
boundary_color = ListedColormap(['#7fe67a', '#f7b081', '#92c4e8'])


# FUNCTIONS ---------------------------------------------------------------------------------------
# Function that, from a list of class labels, returns the list of associate class colors
def get_color_from_class_list(class_list):
    color_list = []
    for class_item in class_list:
        if class_item == 0:
            color_list.append('#0dba04')  # Green
        elif class_item == 1:
            color_list.append('#f0660a')  # Orange
        elif class_item == 2:
            color_list.append('#007bb0')  # Blue
    return color_list


# Function plotting the 2-D graphs for the complete datasets and its partitions (training, validation and test)
# on the base of two features (in_x and in_y) of interest.
def plot_dataset_and_partitions(xtot, ytot, xtrain, ytrain, xvalid, yvalid, xtest, ytest, in_x, in_y, in_size=9):
    in_fig, in_ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    in_ax[0, 0].scatter(xtot[f'{in_x}'], xtot[f'{in_y}'], s=in_size, c=get_color_from_class_list(ytot.values.ravel()))
    in_ax[0, 0].set_title("COMPLETE")
    in_ax[0, 0].set_ylabel(f'{in_y} (y axis)')
    in_ax[0, 1].scatter(xtrain[f'{in_x}'], xtrain[f'{in_y}'], s=in_size, c=get_color_from_class_list(ytrain.values.ravel()))
    in_ax[0, 1].set_title("TRAINING")
    in_ax[1, 0].scatter(xvalid[f'{in_x}'], xvalid[f'{in_y}'], s=in_size, c=get_color_from_class_list(yvalid.values.ravel()))
    in_ax[1, 0].set_title("VALIDATION")
    in_ax[1, 0].set_xlabel(f'{in_x} (x axis)')
    in_ax[1, 0].set_ylabel(f'{in_y} (y axis)')
    in_ax[1, 1].scatter(xtest[f'{in_x}'], xtest[f'{in_y}'], s=in_size, c=get_color_from_class_list(ytest.values.ravel()))
    in_ax[1, 1].set_title("TEST")
    in_ax[1, 1].set_xlabel(f'{in_x} (x axis)')
    plt.tight_layout()
    plt.show()
    return


# Function plotting the 2-D graph of the training points and the decision boundaries
def plot_training_and_boundaries(in_xx, in_yy, in_Z, in_config, in_X_train, in_y_train, in_title, in_parameter, p_name,
                                 in_x, in_y, in_flag=True):

    # Plot of the decision boundaries
    in_fig, in_ax = plt.subplots(figsize=(8, 6))
    plt.pcolormesh(in_xx, in_yy, in_Z, cmap=boundary_color)
    in_fig.suptitle(in_title, fontsize=14, fontweight='bold')

    # Plot of the training points
    plt.scatter(in_X_train[f'{in_x}'], in_X_train[f'{in_y}'], c=get_color_from_class_list(in_y_train.values.ravel()))
    in_ax.set_xlabel(f'{in_x} (x axis)')
    in_ax.set_ylabel(f'{in_y} (y axis)')
    plt.xlim(in_xx.min(), in_xx.max())
    plt.ylim(in_yy.min(), in_yy.max())
    if in_flag:
        plt.title(f"{p_name} = {in_config[in_parameter]}")
    else:
        plt.title(f"{p_name}")
    plt.show()
    return


# Function plotting the accuracy graph for different values of the parameter of interest
def plot_accuracy_plot(x_axis, y_axis, clf_name, param_name, in_flag):
    in_fig, in_ax = plt.subplots(figsize=(8, 6))
    in_fig.suptitle(f"{clf_name} ACCURACY", fontsize=14, fontweight='bold')
    if in_flag:
        plt.xscale('log')  # Logarithmic scale if the parameter is C
    in_ax.plot(x_axis, y_axis, c='#007bb0', linestyle='-', marker='o', label='Accuracy')
    in_ax.set_xlabel(f'{param_name} value')
    in_ax.set_ylabel('Accuracy')
    in_ax.set_title(f"ACCURACY FOR DIFFERENT {param_name} VALUES")
    for in_i, in_j in zip(x_axis, y_axis):  # Plot also the value of the point close to it
        in_ax.annotate(str(in_j), xy=(in_i, in_j))
    plt.show()
    return


# 1 -----------------------------------------------------------------------------------------------
wine_dataset = load_wine()

# First exploration of the dataset
exploration_flag = False
if exploration_flag:
    print(wine_dataset, '\n')  # wine dataset is a dictionary
    print(wine_dataset.keys(), '\n')  # keys are: ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    print("data", '\n', len(wine_dataset['data']), '\n')  # The dataset has 178 items
    print("target", '\n', len(wine_dataset['target']), '\n')
    print("target_names", '\n', len(wine_dataset['target_names']), '\n')  # 13 features
    print("DESCR", '\n', len(wine_dataset['DESCR']), '\n')
    print("feature_names", '\n', len(wine_dataset['feature_names']), '\n')
    print(set(wine_dataset["target"]))  # The classes are: 0, 1, 2

# 2 -----------------------------------------------------------------------------------------------
df = pd.DataFrame(wine_dataset["data"])  # From a dictionary to a Pandas dataframe
df.columns = wine_dataset["feature_names"]
y = pd.DataFrame(wine_dataset["target"])  # Labels column

X = pd.concat([df.alcohol, df.malic_acid], axis=1)  # Dataframe of the only two features of interest

# 3 -----------------------------------------------------------------------------------------------
# Random split of the complete dataset in 5:2:3 Training, Validation and Test
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.3, shuffle=True)
# X_train and y_train are 50% of the dataset <- Training
# X_valid and y_valid are 20% of the dataset <- Validation
# X_test and y_test are 30% of the dataset <- Test

# Plot the complete dataset and the generated partitions
plot_dataset_and_partitions(X, y, X_train, y_train, X_valid, y_valid, X_test, y_test, "alcohol", "malic_acid")

# 4 -----------------------------------------------------------------------------------------------
# K-NEAREST NEIGHBORS -----------------------------------------------------------------------------
# Applying KNN for different K values
hyp_parameters = {
    "n_neighbors": [1, 3, 5, 7]
}

# Decision boundaries plot parameters
h = .02
x_min, x_max = X_train.alcohol.min() - 1, X_train.alcohol.max() + 1
y_min, y_max = X_train.malic_acid.min() - 1, X_train.malic_acid.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Accuracy plot parameters
y_axis_accuracy = []
x_axis_k_value = []
max_accuracy = 0
best_k = 0

for config in ParameterGrid(hyp_parameters):
    clf = KNeighborsClassifier(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    # Plotting decision boundaries and training points
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plot_training_and_boundaries(xx, yy, Z, config, X_train, y_train, "K-NEAREST NEIGHBORS", "n_neighbors", "K",
                                 "alcohol", "malic_acid")

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_k = config["n_neighbors"]  # Keeping track of the parameter value that achieves the best accuracy score

    x_axis_k_value.append(config["n_neighbors"])
    y_axis_accuracy.append(round(Decimal(clf_accuracy), 3))

# 5 -----------------------------------------------------------------------------------------------
# Plotting the accuracy graph
plot_accuracy_plot(x_axis_k_value, y_axis_accuracy, "KNN", "K", False)

# 7 -----------------------------------------------------------------------------------------------
# Training the classifier with the best scoring parameter value on both training and validation set
# and computing the accuracy score on the test set
clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train_valid, y_train_valid.values.ravel())
y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)
print(f"Max accuracy on validation set obtained by K = {best_k}")
print(f"KNN accuracy on test set with first two features = {round(Decimal(clf_accuracy), 3)}", "\n")

# 8 -----------------------------------------------------------------------------------------------
# LINEAR SVC --------------------------------------------------------------------------------------
# Applying LINEAR SVM for different C values
hyp_parameters = {
    "kernel": ["linear"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "max_iter": [1000000]  # Increasing max_iter in order to guarantee convergence
}

# Accuracy plot parameters
y_axis_accuracy = []
x_axis_k_value = []
max_accuracy = 0
best_C = 0

for config in ParameterGrid(hyp_parameters):
    clf = SVC(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    # Plotting decision boundaries and training points
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plot_training_and_boundaries(xx, yy, Z, config, X_train, y_train, "LINEAR SVC", "C", "C", "alcohol", "malic_acid")

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_C = config["C"]  # Keeping track of the parameter value that achieves the best accuracy score

    x_axis_k_value.append(config["C"])
    y_axis_accuracy.append(round(Decimal(clf_accuracy), 3))

# 9 -----------------------------------------------------------------------------------------------
# Plotting the accuracy graph
plot_accuracy_plot(x_axis_k_value, y_axis_accuracy, "LINEAR SVC", "C", True)

# 11 ----------------------------------------------------------------------------------------------
# Training the classifier with the best scoring parameter value on both training and validation set
# and computing the accuracy score on the test set
clf = SVC(C=best_C, kernel="linear", max_iter=1000000)
clf.fit(X_train_valid, y_train_valid.values.ravel())
y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)
print(f"Max accuracy on validation set obtained by C = {best_C}")
print(f"LINEAR SVC accuracy on test set with first two features = {round(Decimal(clf_accuracy), 3)}", "\n")

# 12 ----------------------------------------------------------------------------------------------
# RBF SVM -----------------------------------------------------------------------------------------
hyp_parameters = {
    "kernel": ["rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "max_iter": [1000000]
}

# Accuracy plot parameters
y_axis_accuracy = []
x_axis_k_value = []
max_accuracy = 0
best_C = 0

for config in ParameterGrid(hyp_parameters):
    clf = SVC(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    # Plotting decision boundaries and training points
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plot_training_and_boundaries(xx, yy, Z, config, X_train, y_train, "RBF SVC", "C", "C", "alcohol", "malic_acid")

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_C = config["C"]  # Keeping track of the parameter value that achieves the best accuracy score

    y_axis_accuracy.append(round(Decimal(clf_accuracy), 3))
    x_axis_k_value.append(config["C"])

# Plotting the accuracy graph
plot_accuracy_plot(x_axis_k_value, y_axis_accuracy, "RBF SVC", "C", True)

# 13 ----------------------------------------------------------------------------------------------
# Training the classifier with the best scoring parameter value on both training and validation set
# and computing the accuracy score on the test set
clf = SVC(C=best_C, kernel="rbf", max_iter=1000000)
clf.fit(X_train_valid, y_train_valid.values.ravel())
y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)
print(f"Max accuracy on validation set obtained by C = {best_C}")
print(f"RBF SVC accuracy on test set with first two features = {round(Decimal(clf_accuracy), 3)}", "\n")

# 15 ----------------------------------------------------------------------------------------------
# RBF SVM (C + GAMMA) -----------------------------------------------------------------------------
param_grid = {
    "kernel": ["rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "gamma": [0.005, 0.05, 0.5, 5],
    "max_iter": [1000000]
}

# Accuracy plot parameters
y_axis_accuracy = []
x_axis_k_value = []
max_accuracy = 0
best_config = {}

for config in ParameterGrid(param_grid):
    clf = SVC(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    print(f"RBF SVC without 5-fold: config: {config}, Accuracy: {clf_accuracy}")
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_config = config  # Keeping track of the parameter value that achieves the best accuracy score

clf = SVC(C=best_config["C"], gamma=best_config["gamma"], kernel="rbf", max_iter=1000000)
clf.fit(X_train, y_train.values.ravel())

# Plotting decision boundaries and training points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plot_training_and_boundaries(xx, yy, Z, best_config, X_train, y_train, "RBF SVC", "C",
                             f"C={best_config['C']} Gamma={best_config['gamma']}", "alcohol", "malic_acid", False)

# Training the classifier with the best scoring parameter value on both training and validation set
# and computing the accuracy score on the test set
clf = SVC(C=best_config["C"], gamma=best_config["gamma"], kernel="rbf", max_iter=1000000)
clf.fit(X_train_valid, y_train_valid.values.ravel())
y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)
print("\n", f"Max accuracy on validation set: {round(Decimal(max_accuracy), 3)} obtained by C = {best_config['C']}"
      f" and gamma = {best_config['gamma']}")
print(f"RBF SVC accuracy on test set with first two features = {round(Decimal(clf_accuracy), 3)}", "\n")

# 17 ----------------------------------------------------------------------------------------------
param_grid = {
    "kernel": ["rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "gamma": [0.005, 0.05, 0.5, 5],
    "max_iter": [1000000]
}

cv = StratifiedShuffleSplit(n_splits=5)  # K-Fold Cross-validation
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train_valid, y_train_valid.values.ravel())

scores_dataframe = pd.DataFrame(grid.cv_results_)
print(scores_dataframe[['params', 'mean_test_score']], '\n')

clf = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel="rbf", max_iter=1000000)
clf.fit(X_train, y_train.values.ravel())

# Plotting decision boundaries and training points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plot_training_and_boundaries(xx, yy, Z, grid.best_params_, X_train, y_train, "RBF SVC", "C",
                             f"C={grid.best_params_['C']} Gamma={grid.best_params_['gamma']}", "alcohol", "malic_acid",
                             False)

# 18 ----------------------------------------------------------------------------------------------
# Training the classifier with the best scoring parameter value on both training and validation set
# and computing the accuracy score on the test set
clf = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel="rbf", max_iter=1000000)
clf.fit(X_train_valid, y_train_valid.values.ravel())
y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)
print(f"Max accuracy with cross validation: {round(Decimal(grid.best_score_), 3)} obtained by C = "
      f"{grid.best_params_['C']} and gamma = {grid.best_params_['gamma']}")
print(f"RBF SVC accuracy on test set with first two features = {round(Decimal(clf_accuracy), 3)}", "\n")

# 20 ----------------------------------------------------------------------------------------------
# In order to try different features, instead of choosing them randomly, I'll take the pair of features
# with the best silhouette score and the pair with the worst score, to do so I'll calculate the silhouette
# score for all the pairs of features, avoiding repetitions.
# After that I'll train KNN, LINEAR SVM and RBF SVM on the training set, tuning on the validation set,
# evaluating on the test set and plotting the decision boundaries along with the training points only for the
# most promising value of the parameter.

lists = {
    "x": wine_dataset["feature_names"],
    "y": wine_dataset["feature_names"]
}

max_score = -1
min_score = 1
best_x = ""
best_y = ""
worst_x = ""
worst_y = ""

for config in ParameterGrid(lists):
    if config['x'] != config['y']:
        X = pd.concat([df[f'{config["x"]}'], df[f'{config["y"]}']], axis=1)
        score = silhouette_score(X, y.values.ravel())

        if score > max_score:
            max_score = score
            best_x = config["x"]
            best_y = config["y"]

        if score < min_score:
            min_score = score
            worst_x = config["x"]
            worst_y = config["y"]

# BEST PAIR OF FEATURES ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[f'{best_x}'], df[f'{best_y}'], s=20, c=get_color_from_class_list(y.values.ravel()))
fig.suptitle("BEST FEATURES", fontsize=14, fontweight='bold')
ax.set_title(f"Silhouette_score = {max_score}")
ax.set_xlabel(best_x)
ax.set_ylabel(best_y)
plt.show()

X = pd.concat([df[f'{best_x}'], df[f'{best_y}']], axis=1)
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.3, shuffle=True)

plot_dataset_and_partitions(X, y, X_train, y_train, X_valid, y_valid, X_test, y_test, best_x, best_y, 20)

# Decision boundaries plot parameters
h = .02
x_min, x_max = X_train[f'{best_x}'].min() - 1, X_train[f'{best_x}'].max() + 1
y_min, y_max = X_train[f'{best_y}'].min() - 1, X_train[f'{best_y}'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# KNN ---------------------------------------------------------------------------------------------
hyp_parameters = {
    "n_neighbors": [1, 3, 5, 7]
}

# Accuracy plot parameters
max_accuracy = 0
best_k = 0

for config in ParameterGrid(hyp_parameters):
    clf = KNeighborsClassifier(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_k = config["n_neighbors"]  # Keeping track of the parameter value that achieves the best accuracy score

clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train_valid, y_train_valid.values.ravel())

# Plotting decision boundaries and training points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plot_training_and_boundaries(xx, yy, Z, {"n_neighbors": best_k}, X_train_valid, y_train_valid, "K-NEAREST NEIGHBORS",
                             "n_neighbors", "K", best_x, best_y)

y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)  # Computing accuracy on test set
print(f"KNN accuracy on test set with best features = {round(Decimal(clf_accuracy), 3)}", "\n")

# LINEAR SVC --------------------------------------------------------------------------------------
hyp_parameters = {
    "kernel": ["linear"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "max_iter": [1000000]  # Increasing max_iter in order to guarantee convergence
}

# Accuracy plot parameters
max_accuracy = 0
best_C = 0

for config in ParameterGrid(hyp_parameters):
    clf = SVC(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_C = config["C"]  # Keeping track of the parameter value that achieves the best accuracy score

clf = SVC(C=best_C, kernel="linear", max_iter=1000000)
clf.fit(X_train_valid, y_train_valid.values.ravel())

# Plotting decision boundaries and training points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plot_training_and_boundaries(xx, yy, Z, {"C": best_C}, X_train_valid, y_train_valid, "LINEAR SVC", "C", "C", best_x,
                             best_y)

y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)  # Computing accuracy on test set
print(f"LINEAR SVC accuracy on test set with best features = {round(Decimal(clf_accuracy), 3)}", "\n")

# RBF SVM -----------------------------------------------------------------------------------------
hyp_parameters = {
    "kernel": ["rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "max_iter": [1000000]  # Increasing max_iter in order to guarantee convergence
}

# Accuracy plot parameters
max_accuracy = 0
best_C = 0

for config in ParameterGrid(hyp_parameters):
    clf = SVC(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_C = config["C"]  # Keeping track of the parameter value that achieves the best accuracy score

clf = SVC(C=best_C, kernel="rbf", max_iter=1000000)
clf.fit(X_train_valid, y_train_valid.values.ravel())

# Plotting decision boundaries and training points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plot_training_and_boundaries(xx, yy, Z, {"C": best_C}, X_train_valid, y_train_valid, "RBF SVC", "C",
                             "C", best_x, best_y)

y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)  # Computing accuracy on test set
print(f"RBF SVM accuracy on test set with best features = {round(Decimal(clf_accuracy), 3)}", "\n")

# WORST PAIR OF FEATURES --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[f'{worst_x}'], df[f'{worst_y}'], s=20, c=get_color_from_class_list(y.values.ravel()))
fig.suptitle("WORST FEATURES", fontsize=14, fontweight='bold')
ax.set_title(f"Silhouette_score = {min_score}")
ax.set_xlabel(worst_x)
ax.set_ylabel(worst_y)
plt.show()

X = pd.concat([df[f'{worst_x}'], df[f'{worst_y}']], axis=1)
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.3, shuffle=True)

plot_dataset_and_partitions(X, y, X_train, y_train, X_valid, y_valid, X_test, y_test, worst_x, worst_y, 20)

# Decision boundaries plot parameters
h = .02
x_min, x_max = X_train[f'{worst_x}'].min() - 1, X_train[f'{worst_x}'].max() + 1
y_min, y_max = X_train[f'{worst_y}'].min() - 1, X_train[f'{worst_y}'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# KNN ---------------------------------------------------------------------------------------------
hyp_parameters = {
    "n_neighbors": [1, 3, 5, 7]
}

# Accuracy plot parameters
max_accuracy = 0
best_k = 0

for config in ParameterGrid(hyp_parameters):
    clf = KNeighborsClassifier(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_k = config["n_neighbors"]  # Keeping track of the parameter value that achieves the best accuracy score

clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train_valid, y_train_valid.values.ravel())

# Plotting decision boundaries and training points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plot_training_and_boundaries(xx, yy, Z, {"n_neighbors": best_k}, X_train_valid, y_train_valid, "K-NEAREST NEIGHBORS",
                             "n_neighbors", "K", worst_x, worst_y)

y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)  # Computing accuracy on test set
print(f"KNN accuracy on test set with worst features = {round(Decimal(clf_accuracy), 3)}", "\n")

# LINEAR SVC --------------------------------------------------------------------------------------
hyp_parameters = {
    "kernel": ["linear"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "max_iter": [1000000]  # Increasing max_iter in order to guarantee convergence
}

# Accuracy plot parameters
max_accuracy = 0
best_C = 0

for config in ParameterGrid(hyp_parameters):
    clf = SVC(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_C = config["C"]  # Keeping track of the parameter value that achieves the best accuracy score

clf = SVC(C=best_C, kernel="linear", max_iter=1000000)
clf.fit(X_train_valid, y_train_valid.values.ravel())

# Plotting decision boundaries and training points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plot_training_and_boundaries(xx, yy, Z, {"C": best_C}, X_train_valid, y_train_valid, "LINEAR SVC", "C", "C", worst_x,
                             worst_y)

y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)  # Computing accuracy on test set
print(f"LINEAR SVC accuracy on test set with worst features = {round(Decimal(clf_accuracy), 3)}", "\n")

# RBF SVM -----------------------------------------------------------------------------------------
hyp_parameters = {
    "kernel": ["rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "max_iter": [1000000]  # Increasing max_iter in order to guarantee convergence
}

# Accuracy plot parameters
max_accuracy = 0
best_C = 0

for config in ParameterGrid(hyp_parameters):
    clf = SVC(**config)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_valid)

    clf_accuracy = accuracy_score(y_valid, y_pred)  # Accuracy on the validation set
    if clf_accuracy > max_accuracy:
        max_accuracy = clf_accuracy
        best_C = config["C"]  # Keeping track of the parameter value that achieves the best accuracy score

clf = SVC(C=best_C, kernel="rbf", max_iter=1000000)
clf.fit(X_train_valid, y_train_valid.values.ravel())

# Plotting decision boundaries and training points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plot_training_and_boundaries(xx, yy, Z, {"C": best_C}, X_train_valid, y_train_valid, "RBF SVC", "C",
                             "C", worst_x, worst_y)

y_pred = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, y_pred)  # Computing accuracy on test set
print(f"RBF SVM accuracy on test set with worst features = {round(Decimal(clf_accuracy), 3)}", "\n")

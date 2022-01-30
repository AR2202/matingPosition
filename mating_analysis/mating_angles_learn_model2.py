import numpy as np
import os
import re
import sklearn
import sklearn.preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import mating_angles_model2
from mating_angles_model2 import filtered_outputs, unfiltered_outputs
from mating_angles_model2 import load_csv_file, tilting_index, tilting_index_all_frames
from sklearn.linear_model import SGDClassifier, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from joblib import dump, load
# This is the machine learning model for DLC model2
# expects the data to be created by a DLC model that labels both flies' Head
# and Abdomen as well as the male's Shoulders

# Feature scaling


def scale_angles(angles):
    """scales the input array to have values between 0 and 1."""
    angles_matrix = angles.values.reshape(-1, 1)
    scaled = sklearn.preprocessing.MinMaxScaler()
    scaled_angles = scaled.fit_transform(angles_matrix)
    return(scaled_angles)


def scale_filtered(path,
                   P,
                   removeWall=False,
                   minWallDist=3,
                   copstartframe=500):
    """loads the csv file of deeplabcut data
    specified as the path argument and determines mating angle
    from both wing and body axis data;
    returns the angles based on wing data and the angles based on body axis
    (in this order);
    scales the data
    This is the function that should be used if you want filtering of data by 
    those with a likelihood > P"""
    angles_b, wing_dist_male, abd_dist, head_dist, rownumbers =\
        filtered_outputs(path, P, removeWall=removeWall,
                         minWallDist=minWallDist)
    angles_b_scaled = scale_angles(angles_b)
    wing_dist_male_scaled = scale_angles(wing_dist_male)
    abd_dist_scaled = scale_angles(abd_dist)
    head_dist_scaled = scale_angles(head_dist)
    tilting_index_scaled = scale_angles(
        tilting_index_all_frames(
            wing_dist_male,
            copstartframe,
            rownumbers=rownumbers))
    return angles_b_scaled, wing_dist_male_scaled, abd_dist_scaled, head_dist_scaled, tilting_index_scaled


def scale_unfiltered(path, removeWall=False, minWallDist=3, copstartframe=500):
    """loads the csv file of deeplabcut data
    specified as the path argument and determines mating angle
    from both wing and body axis data;
    returns the angles based on wing data and the angles based on body axis
    (in this order);
    scales the data
    This is the function that should be used if you don't want filtering of data """
    angles_b, wing_dist_male, abd_dist, head_dist, rownumbers =\
        unfiltered_outputs(path,
                           removeWall=removeWall,
                           minWallDist=minWallDist)
    angles_b_scaled = scale_angles(angles_b)
    wing_dist_male_scaled = scale_angles(wing_dist_male)
    abd_dist_scaled = scale_angles(abd_dist)
    head_dist_scaled = scale_angles(head_dist)
    tilting_index_scaled = scale_angles(tilting_index_all_frames(
                                        wing_dist_male, copstartframe,
                                        rownumbers=rownumbers))
    return angles_b_scaled, wing_dist_male_scaled, abd_dist_scaled, head_dist_scaled, tilting_index_scaled

# Classification models


def train_SGD(X, y, loss="log"):
    """trains the stochastic gradient descent algorithm
    with default loss="log" it uses logistic regression
    uses anova chi2 for feature selection"""
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    n = X_train.shape[0]
    max_iter = np.ceil(10**6 / n)
    parameters = {
        'sgd__alpha': (0.00001, 0.000001),
        'sgd__penalty': ('l2', 'elasticnet'),
    }
    pipe = Pipeline([('anova', SelectPercentile(f_classif)),
                     ('sgd', SGDClassifier(
                         loss=loss,
                         max_iter=max_iter,
                         early_stopping=True))])
    grid_search = GridSearchCV(pipe, parameters, verbose=1)
    clf = grid_search.fit(X_train, y_train)
    CVScore = grid_search.best_score_
    testScore = clf.score(X_test, y_test)
    return clf, testScore, CVScore


def train_knn(X, y):
    """trains the k nearest neighbors algoithm
    feature selection is done by support vector classifier"""
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    parameters = {'knc__n_neighbors': [1, 2, 3, 4, 5],
                  'knc__weights': ['uniform', 'distance']}
    pipe = Pipeline([('feature_selection', SelectFromModel(LinearSVC())),
                    ('knc', KNeighborsClassifier())])
    grid_search = GridSearchCV(pipe, parameters, verbose=1)
    knn = grid_search.fit(X_train, y_train)
    CVScore = grid_search.best_score_
    testScore = knn.score(X_test, y_test)
    return knn, testScore, CVScore


def train_SVC(X, y):
    """trains the support vector classifier
    feature selction is done with anova"""
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    param_grid = [
        {'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['linear']},
        {'svc__C': [1, 10, 100, 1000], 'svc__gamma': [0.001, 0.0001],
         'svc__kernel': ['rbf']},
    ]
    pipe = Pipeline([('anova', SelectPercentile(f_classif)),
                     ('svc', SVC(probability=True))])
    grid_search = GridSearchCV(pipe, param_grid, verbose=1)
    supportV = grid_search.fit(X_train, y_train)
    CVScore = grid_search.best_score_
    testScore = supportV.score(X_test, y_test)
    return supportV, testScore, CVScore


def train_NB(X, y):
    """trains the naive bayes algorithm;
    feature selection is done with linear support vecotor classifier"""
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipe = Pipeline([('feature_selection', SelectFromModel(LinearSVC())),
                    ('classification', GaussianNB())])
    NB = pipe.fit(X_train, y_train)
    testScore = NB.score(X_test, y_test)
    return NB, testScore


def train_randomForest(X, y):
    """trains the randomForst algorithm;
    feature selection by support vector classifier"""
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    parameters = {'rfc__n_estimators': [10, 25, 50, 75, 100, 125]}
    pipe = Pipeline([('feature_selection', SelectFromModel(LinearSVC())),
                    ('rfc', RandomForestClassifier())])
    grid_search = GridSearchCV(pipe, parameters, verbose=1)
    randomF = grid_search.fit(X_train, y_train)
    CVScore = grid_search.best_score_
    testScore = randomF.score(X_test, y_test)
    return randomF, testScore, CVScore


def prepare_training_data(path, filtering=False, P=0.8, copstartframe=500,
                          removeWall=False,
                          minWallDist=3,
                          featurelist=["angles_b_scaled",
                                       "head_dist_scaled",
                                       "abd_dist_scaled",
                                       "tilting_index_scaled"]):
    """loads csv file and scales the features, then makes an np.array
    of the features and returns the array"""
    X = np.array([])
    if filtering:
        angles_b_scaled, wing_dist_male_scaled, abd_dist_scaled, head_dist_scaled, tilting_index_scaled =\
            scale_filtered(path, P, copstartframe=copstartframe,
                           removeWall=removeWall, minWallDist=minWallDist)
    else:
        angles_b_scaled, wing_dist_male_scaled, abd_dist_scaled, head_dist_scaled, tilting_index_scaled =\
            scale_unfiltered(path, copstartframe=copstartframe,
                             removeWall=removeWall, minWallDist=minWallDist)
    # tilting_index=tilting_index_all_frames(wing_dist_male_scaled,
    # wing_dist_female_scaled, copstartframe)
    for feature in featurelist:
        if X.size > 0:
            X = np.append(X, eval(feature), axis=1)
        else:
            X = eval(feature)
    return X


def import_train_test(path_to_csv, path_to_images, positives, filtering=False,
                      P=0.8, copstartframe=500,
                      removeWall=False,
                      minWallDist=3,
                      featurelist=["angles_b_scaled",
                                   "tilting_index_scaled",
                                   "head_dist_scaled",
                                   "abd_dist_scaled"]):
    """prepares training dataset"""
    """if positives is a list of framenumbers, the first frame should be 1;
    uses a directory with the images as input;
    unlikely to be used in future"""
    X = prepare_training_data(path_to_csv, filtering=filtering, P=P,
                              featurelist=featurelist,
                              copstartframe=copstartframe,
                              removeWall=removeWall,
                              minWallDist=minWallDist)
    num = [int(re.search('d+', filename).group(0)) for filename
           in os.listdir(path_to_images)]
    num_shifted = [numb-1 for numb in num]
    X_training = X[num_shifted]
    y_training = [0 for i in num_shifted]
    positives_shifted = [pos-1 for pos in positives]
    for pos in positives_shifted:
        y_training[pos] = 1
    return X_training, y_training


def import_train_test_from_csv(paths_to_csv, paths_to_labels, filtering=False,
                               P=0.8,
                               removeWall=False,
                               minWallDist=3,
                               featurelist=["angles_b_scaled",
                                            "tilting_index_scaled",
                                            "head_dist_scaled",
                                            "abd_dist_scaled"]):
    """prepares training dataset from a csv file of labelled frames"""
    copstartframes = []
    Xs_training = np.array([])
    ys_training = np.array([])
    for path_to_csv, path_to_labels in zip(paths_to_csv, paths_to_labels):
        labeltable = pd.read_csv(path_to_labels, header=0)
        copstartframe = int(labeltable[labeltable.keys()[0]][0])
        nums_neg = []
        nums_pos = []
        X = prepare_training_data(path_to_csv, filtering=filtering, P=P,
                                  featurelist=featurelist,
                                  copstartframe=copstartframe,
                                  removeWall=removeWall,
                                  minWallDist=minWallDist)
        for i in range(0, len(labeltable[labeltable.keys()[1]]), 2):
            nums_neg = nums_neg+list(range(labeltable[labeltable.keys()[1]][i],
                                           labeltable[labeltable.keys()[1]][i+1]))
        for i in range(0, len(labeltable[labeltable.keys()[2]]), 2):
            nums_pos = nums_pos+list(range(labeltable[labeltable.keys()[2]][i],
                                           labeltable[labeltable.keys()[2]][i+1]))
        nums = nums_neg+nums_pos
        y_neg = np.zeros(len(nums_neg), int)
        y_pos = np.ones(len(nums_pos), int)
        X_training = X[nums]
        y_training = np.concatenate([y_neg, y_pos])
        ys_training = np.concatenate([ys_training, y_training])
        if Xs_training.size > 0:
            Xs_training = np.concatenate([Xs_training, X_training])
        else:
            Xs_training = X_training
        copstartframes.append(copstartframe)
    return Xs_training, ys_training, copstartframes


def learning_pipeline(paths_to_csv, paths_to_images, positives=[],
                      training_only=True, filtering_data=False,
                      filtering_train=False, P=0.8,
                      copstartframe=500,
                      training_from_csv=True,
                      filename='trained_models.joblib',
                      removeWall=False,
                      minWallDist=3,
                      featurelist=["angles_b_scaled",
                                   "head_dist_scaled",
                                   "abd_dist_scaled",
                                   "tilting_index_scaled"]):
    """pipeline for machine learning;
    prepares the training dataset, trains all models, tests all models;
    outputs the models as a dictionary"""
    if training_from_csv:
        X, y, copstartframes = import_train_test_from_csv(paths_to_csv,
                                                          paths_to_images,
                                                          filtering=filtering_train,
                                                          P=P,
                                                          removeWall=False,
                                                          # should be false to
                                                          # avoid shift in labels
                                                          minWallDist=minWallDist,
                                                          featurelist=featurelist)
        copstartframe = copstartframes[0]
    else:
        X, y = import_train_test(paths_to_csv[0], paths_to_images[0],
                                 positives,
                                 filtering=filtering_train,
                                 P=P,
                                 removeWall=False,
                                 minWallDist=minWallDist,
                                 featurelist=featurelist,
                                 copstartframe=copstartframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    logReg, logRegScore, logRegCVScore = train_SGD(
        X_train, y_train, loss="log")
    print("Logistic Regression Test Score: {}".format(logRegScore))
    print("Logistic Regression CV Score: {}".format(logRegCVScore))
    suppVC, SVCScore, SVCCVScore = train_SVC(X_train, y_train)
    print("Support Vector Machine Test Score: {}".format(SVCScore))
    print("Support Vector Machine CV Score: {}".format(SVCCVScore))
    knn, knnScore, knnCVScore = train_knn(X_train, y_train)
    print("K Nearest Neighbors Test Score: {}".format(knnScore))
    print("K Nearest Neighbors CV Score: {}".format(knnCVScore))
    randomF, randomFScore, randomFCVScore = train_randomForest(X_train,
                                                               y_train)
    print("Random Forest Test Score: {}".format(randomFScore))
    print("Random Forest CV Score: {}".format(randomFCVScore))
    NB, NBScore = train_NB(X_train, y_train)
    print("Naive Bayes Test Score: {}".format(NBScore))
    # evaluation of the ensemble model
    predLogReg = logReg.predict_proba(X_test)
    predSVC = suppVC.predict_proba(X_test)
    predKnn = knn.predict_proba(X_test)
    predRandomF = randomF.predict_proba(X_test)
    predNB = NB.predict_proba(X_test)
    # creating predictions by averaging the predicted
    # class probabilities from each model
    ensemblePred = (predLogReg+predSVC+predKnn+predRandomF+predNB)/5
    yProb = ensemblePred[:, 1]
    yPred = np.apply_along_axis(np.argmax, 1, ensemblePred)
    accuracy = accuracy_score(y_test, yPred)
    balanced_accuracy = balanced_accuracy_score(y_test, yPred)
    f1 = f1_score(y_test, yPred)
    logloss = log_loss(y_test, yProb)
    rocauc = roc_auc_score(y_test, yProb)
    print("Ensemble Model Accuracy Score: {}".format(accuracy))
    print("Ensemble Model Balanced Accuracy Score: {}".format(balanced_accuracy))
    print("Ensemble Model F1 Score: {}".format(f1))
    print("Ensemble Model Log Loss Score: {}".format(logloss))
    print("Ensemble Model ROC AUC Score: {}".format(rocauc))

    if training_only:
        models = {
            "LogReg": {"model": logReg,
                       "score": logRegScore,
                       "CVScore": logRegCVScore},
            "SVC": {"model": suppVC,
                    "score": SVCScore,
                    "CVScore": SVCCVScore},
            "KNN": {"model": knn,
                    "score": knnScore,
                    "CVScore": knnCVScore},
            "RFC": {"model": randomF,
                    "score": randomFScore,
                    "CVScore": randomFCVScore},
            "NB": {"model": NB,
                   "score": NBScore},
            "ensemble": {"accuracy": accuracy,
                         "F1": f1,
                         "LogLoss": logloss,
                         "ROCAUC": rocauc}
        }
    else:
        # making predictions for the data
        data = prepare_training_data(paths_to_csv[0],
                                     filtering=filtering_data,
                                     P=P,
                                     removeWall=removeWall,
                                     minWallDist=minWallDist,
                                     featurelist=featurelist,
                                     copstartframe=copstartframe)
        predictionsLogReg = logReg.predict_proba(data)
        predictionsSVC = suppVC.predict_proba(data)
        predictionsKnn = knn.predict_proba(data)
        predictionsRandomF = randomF.predict_proba(data)
        predictionsNB = NB.predict_proba(data)
        # creating predictions by averaging the predicted class probabilities
        # from each model
        ensembePredictions = (predictionsLogReg + predictionsSVC
                              + predictionsKnn
                              + predictionsRandomF
                              + predictionsNB)/5
        classPredictions = np.apply_along_axis(np.argmax, 1,
                                               ensembePredictions)
        models = {
            "LogReg": {"model": logReg,
                       "score": logRegScore,
                       "CVScore": logRegCVScore,
                       "predictions": predictionsLogReg},
            "SVC": {"model": suppVC,
                    "score": SVCScore,
                    "CVScore": SVCCVScore,
                    "predictions": predictionsSVC},
            "KNN": {"model": knn,
                    "score": knnScore,
                    "CVScore": knnCVScore,
                    "predictions": predictionsKnn},
            "RFC": {"model": randomF,
                    "score": randomFScore,
                    "CVScore": randomFCVScore,
                    "predictions": predictionsRandomF},
            "NB": {"model": NB,
                   "score": NBScore,
                   "predictions": predictionsNB},
            "ensemble": {"predictions": ensembePredictions,
                         "classPredictions": classPredictions,
                         "accuracy": accuracy,
                         "F1": f1,
                         "LogLoss": logloss,
                         "ROCAUC": rocauc}
        }
    dump(models, filename)
    return models


def load_pretrained(filename='trained_models.joblib'):
    """reload the pretrained model"""
    models = load(filename)
    return models


def apply_pretrained(models, data, startframe=0):
    """apply the pretrained model to new data"""
    """startframe can be used to subset data -
    for example to include only copulation frames"""
    # load models
    logReg = models["LogReg"]["model"]
    suppVC = models["SVC"]["model"]
    knn = models["KNN"]["model"]
    randomF = models["RFC"]["model"]
    NB = models["NB"]["model"]
    # predict data
    predictionsLogReg = logReg.predict_proba(data)
    predictionsSVC = suppVC.predict_proba(data)
    predictionsKnn = knn.predict_proba(data)
    predictionsRandomF = randomF.predict_proba(data)
    predictionsNB = NB.predict_proba(data)
    ensembePredictions = (predictionsLogReg+predictionsSVC+predictionsKnn
                          + predictionsRandomF + predictionsNB)/5
    classPredictions = np.apply_along_axis(np.argmax, 1, ensembePredictions)
    classPredictions = classPredictions[startframe:]
    fraction_positives = len(classPredictions[classPredictions == 1])/len(
        classPredictions)
    return classPredictions, fraction_positives


def evalulate_pretrained(paths_to_csv, paths_to_images, positives=[],
                         copstartframe=500, testdata_from_csv=True,
                         filename='trained_models.joblib',
                         removeWall=False,
                         minWallDist=3,
                         featurelist=["angles_b_scaled",
                                      "head_dist_scaled",
                                      "tilting_index_scaled",
                                      "abd_dist_scaled"]):
    """evaluates a pretrained model on new test data"""
    # load models
    models = load_pretrained(filename=filename)
    logReg = models["LogReg"]["model"]
    suppVC = models["SVC"]["model"]
    knn = models["KNN"]["model"]
    randomF = models["RFC"]["model"]
    NB = models["NB"]["model"]
    if testdata_from_csv:
        X_test, y_test, copstartframe = import_train_test_from_csv(
            paths_to_csv, paths_to_images,
            filtering=False, P=0.8,
            removeWall=removeWall,
            minWallDist=minWallDist,
            featurelist=featurelist)
    else:
        X_test, y_test = import_train_test(
            paths_to_csv,
            paths_to_images,
            positives,
            filtering=False,
            P=0.8,
            removeWall=removeWall,
            minWallDist=minWallDist,
            featurelist=featurelist,
            copstartframe=copstartframe)
    # evaluation of the ensemble model
    predLogReg = logReg.predict_proba(X_test)
    predSVC = suppVC.predict_proba(X_test)
    predKnn = knn.predict_proba(X_test)
    predRandomF = randomF.predict_proba(X_test)
    predNB = NB.predict_proba(X_test)
    # creating predictions by averaging
    # the predicted class probabilities from each model
    ensemblePred = (predLogReg+predSVC+predKnn+predRandomF+predNB)/5
    yProb = ensemblePred[:, 1]
    yPred = np.apply_along_axis(np.argmax, 1, ensemblePred)
    accuracy = accuracy_score(y_test, yPred)
    balanced_accuracy = balanced_accuracy_score(y_test, yPred)
    f1 = f1_score(y_test, yPred)
    logloss = log_loss(y_test, yProb)
    rocauc = roc_auc_score(y_test, yProb)
    print("Ensemble Model Accuracy Score: {:.2f}".format(accuracy))
    print("Ensemble Model Balanced Accuracy Score: {:.2f}".format(
        balanced_accuracy))
    print("Ensemble Model F1 Score: {:.2f}".format(f1))
    print("Ensemble Model Log Loss Score: {:.2f}".format(logloss))
    print("Ensemble Model ROC AUC Score: {:.2f}".format(rocauc))
    scores = {"accuracy": accuracy,
              "balanced_accuracy": balanced_accuracy,
              "F1": f1,
              "LogLoss": logloss,
              "ROCAUC": rocauc}
    return scores

# Regression Models
# This model can be used to infer the position of a body part
# based on the position of the other body parts
# Used to label points that were labelled with low
# likelihood in DLC


def train_RidgeRegressor(X, y):
    """trains the ridge regression model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    ridge = RidgeCV()
    reg = ridge.fit(X_train, y_train)
    testScore = reg.score(X_test, y_test)
    y_predicted = reg.predict(X_test)
    plt.figure()
    plt.scatter(y_test, y_predicted)
    plt.show()
    return reg, testScore

# prepare training data


def coords_to_1D(df_x, df_y):
    """transforms the coordinates to a one dimensional array in one row of the dataframe"""
    return (600*df_y+df_x)


def prepare_features(headX, headY, abdomenX, abdomenY, wing1X, wing1Y, wing2X, wing2Y):
    """prepares the features of the regression model"""
    head = scale_angles(coords_to_1D(headX, headY))
    abdomen = scale_angles(coords_to_1D(abdomenX, abdomenY))
    wing1 = scale_angles(coords_to_1D(wing1X, wing1Y))
    wing2 = scale_angles(coords_to_1D(wing2X, wing2Y))
    features = [head, abdomen, wing1, wing2]
    return features


def prepare_male_features(path):
    """prepares the features for male data of a csv file"""
    data = load_csv_file(path)
    malefeatures = prepare_features(data.MaleHeadX,
                                    data.MaleHeadY,
                                    data.MaleAbdomenX,
                                    data.MaleAbdomenY,
                                    data.MaleLEftShoulderX,
                                    data.MaleLEftShoulderY,
                                    data.MaleRightShoulderX,
                                    data.MaleRightShoulderY)

    return malefeatures


def prepare_regression_training_features(path, label=1):
    """prepares the features for training a regression model;
    label: the index of the feature that is used as a label for training purposes;
    corresponds to the index of features in prepare_features()
    0:head
    1:abdomen
    2:wing1
    3:wing2
    """
    malefeatures = prepare_male_features(path)
    labelMale = malefeatures[label]
    del malefeatures[label]
    malefeat = np.concatenate(malefeatures, axis=1)
    return malefeat, labelMale


def train_regression_models(path, label=1):
    """label: the index of the feature that is used as a label for training purposes;
    corresponds to the index of features in prepare_features()
    0:head
    1:abdomen
    2:wing1
    3:wing2
    """
    Xm, ym = prepare_regression_training_features(path, label=label)
    maleRidge, maleRidgeScore = train_RidgeRegressor(Xm, ym)
    print("male Ridge Regression Score: {}".format(maleRidgeScore))

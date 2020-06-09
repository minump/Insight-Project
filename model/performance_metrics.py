import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
import pandas as pd
#from sklearn.metrics import roc_curve
from matplotlib import pyplot


def get_metrics(true_labels, predicted_labels):
    accuracy = np.round(metrics.accuracy_score(true_labels, predicted_labels), 4)
    prec = np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 4)
    recall = np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 4)
    f1 = np.round( metrics.f1_score(true_labels, predicted_labels, average='weighted'), 4)

    df = pd.DataFrame([[accuracy, prec, recall, f1]], index=['performance'], columns=["accuracy", "precision", "recall", "f1_score"])
    print(df)
    return df


# get AUC score for both train and test data sets

def roc_auc_scorer( y, y_pred ) :
    roc_score = metrics.roc_auc_score( y, y_pred )
    print("roc_score ",roc_score)
    lr_fpr, lr_tpr, _ = roc_curve(y, y_pred)
    
    # plot the roc curve for the model
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='XGBoost')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

#roc_auc_scorer( clf, x_train, y_train ), roc_auc_scorer( clf, x_test, y_test )  # (0.7510190300457636, 0.6801152652902591)

def display_model_performance_metrics(true_labels, predicted_labels, predicted_proba, classes=[1, 0]):
    print('Model Performance metrics:')
    print('-' * 30)
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    
    print('\nROC plot')
    roc_auc_scorer(true_labels, predicted_proba)



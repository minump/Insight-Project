import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
import streamlit as st
from matplotlib import pyplot


def get_metrics(true_labels, predicted_labels):
    accuracy = np.round(metrics.accuracy_score(true_labels, predicted_labels), 4)
    prec = np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 4)
    recall = np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 4)
    f1 = np.round( metrics.f1_score(true_labels, predicted_labels, average='weighted'), 4)

    metrics_df = pd.DataFrame([[accuracy, prec, recall, f1]], index=['performance'], columns=["Avg accuracy", "Avg precision", "Avg recall", "Avg f1_score"])
    
    return metrics_df 


def get_classification_report(true_labels, predicted_labels, classes):
    report = metrics.classification_report(true_labels, predicted_labels, labels=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df


def get_confusion_matrix(true_labels, predicted_labels, classes):
    total_classes = len(classes)
    level_labels = [total_classes * [0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, labels=classes)

    cm_df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(classes):
        rowdata={}
        # columns
        for j, col_label in enumerate(classes): 
            rowdata[col_label]=cm[i,j]
        cm_df = cm_df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    
    return cm_df[classes]


    

def roc_auc_scorer( y, y_pred ) :
    roc_score = metrics.roc_auc_score( y, y_pred )
    print("roc_score ",roc_score)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(y, y_pred)
    
    # plot the roc curve for the model
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='XGBoost')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    
def show_precision_recall_curve(model, model_title, x_test, y_test):
    disp = metrics.plot_precision_recall_curve(model, x_test, y_test)
    disp.ax_.set_title(str(model_title)+' Precision-Recall curve ')
    # show the plot
    st.pyplot()

def show_confusion_matrix(model, model_title, x_test, y_test):
    disp = metrics.plot_confusion_matrix(model, x_test, y_test)
    disp.ax_.set_title(str(model_title)+' Confusion matrix plot')
    # show the plot
    st.pyplot()
    


def display_model_performance_metrics(model, model_title, x_test, true_labels, predicted_labels, predicted_proba, classes):
    print('Model Performance metrics:')
    print('-' * 30)
    st.write('-' * 30)
    st.write('Model Performance metrics:')
    st.write('Tested on  data : Number of samples = ', x_test.shape[0], 
             ' Number of features = ', x_test.shape[1])
    st.write('-' * 30)
    metrics_df = get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    print(metrics_df)
    print('Model classification report :')
    print('-'*30)
    report_df = get_classification_report(true_labels, predicted_labels, classes)
    print(report_df)
    st.write('Model classification report :')
    st.dataframe(report_df)
    print('Prediction confusion matrix :')
    cm_frame = get_confusion_matrix(true_labels, predicted_labels, classes)
    print(cm_frame)
    st.write('Prediction confusion matrix :')
    st.dataframe(cm_frame)
    show_precision_recall_curve(model, model_title, x_test, true_labels)
    show_confusion_matrix(model, model_title, x_test, true_labels)
    
    
    



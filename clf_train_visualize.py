import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

plt.rcParams['font.sans-serif'] = ['Microsoft Yahei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False



def classifier(model, x_train, x_test, y_train, y_test, train_dict={}, test_dict={}, model_name='', plot_CM=True, plot_ROC=True):
    
    model.fit(x_train,y_train)
    y_train_pred=model.predict(x_train)
    y_test_pred=model.predict(x_test)
    prob_y_train=model.predict_proba(x_train)[:,1]
    prob_y_test=model.predict_proba(x_test)[:,1]
    
    if plot_CM:
        cm_train=confusion_matrix(y_train,y_train_pred)
        cm_test=confusion_matrix(y_test,y_test_pred)
        plot_ConfusionMatrix(model, cm_train, cm_test)
        
    if plot_ROC:
        plot_ROC_curve(model, y_train, y_test, prob_y_train, prob_y_test)

    if len(model_name)==0:
        model_name=str(model)

    train_dict[model_name+'+accuracy_score']=accuracy_score(y_train, y_train_pred)
    test_dict[model_name+'+accuracy_score']=accuracy_score(y_test, y_test_pred)
    train_dict[model_name+'+auc']=roc_auc_score(y_train, prob_y_train)
    test_dict[model_name+'+auc']=roc_auc_score(y_test, prob_y_test)

    
    return y_train_pred, y_test_pred, prob_y_train, prob_y_test, train_dict, test_dict



def plot_ConfusionMatrix(model, cm_train, cm_test):
    
    disp_train=ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_test=ConfusionMatrixDisplay(confusion_matrix=cm_test)

    fig,axs=plt.subplots(nrows=1,ncols=2,figsize=(18,6))

    disp_train.plot(ax=axs[0],cmap='Greens')
    axs[0].set_title('Train CM + '+str(model))

    disp_test.plot(ax=axs[1],cmap='Blues')
    axs[1].set_title('Test CM + '+str(model))

    plt.tight_layout()
    plt.show()



def plot_ROC_curve(model, y_train, y_test, prob_y_train, prob_y_test):

    auc_train=roc_auc_score(y_train,prob_y_train)
    auc_test=roc_auc_score(y_test,prob_y_test)

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train,prob_y_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test,prob_y_test)

    plt.figure(figsize=(18,6))

    plt.subplot(1,2,1)
    plt.plot(fpr_train, tpr_train, 'g', lw=2, label='Train ROC curve (area=%0.4f)' % auc_train)
    plt.plot([0,1],[0,1],'k-.')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Train ROC curve + '+str(model))
    plt.legend(loc='lower right')

    plt.subplot(1,2,2)
    plt.plot(fpr_test, tpr_test, 'b', lw=2, label='Test ROC curve (area=%0.4f)' % auc_test)
    plt.plot([0,1],[0,1],'k-.')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Test ROC curve + '+str(model))
    plt.legend(loc='lower right')

    plt.show()
    
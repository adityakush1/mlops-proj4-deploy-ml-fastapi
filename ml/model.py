from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import sys
import os
from datetime import datetime

import pandas as pd
import multiprocessing
import seaborn as sns
from matplotlib import pyplot as plt




current_dir = os.path.dirname(os.path.abspath(__file__))

# Adding parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

#print(sys.path)
from general_logger import get_general_logger
from general_config import IMAGE_PATH




date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = get_general_logger('model', date_str)

logger.debug('Starting')


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    using GridSearchCV for hyperparameter tuning and cross validation

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    parameters = { 'n_estimators': [10,15,20, 30],
                  'max_depth': [5,10],
                  'min_samples_split': [25, 50, 100],
                  'learning_rate': [0.1, 0.3, 0.5]                  
                  }
    
    n_jobs = multiprocessing.cpu_count()-1
    logger.info(f"Search best hyperparamets on {n_jobs} cores")
    
    gb_clf = GridSearchCV(GradientBoostingClassifier(random_state=0), 
                          param_grid=parameters,
                          cv=5,
                          verbose=2
                          )

    logger.debug("Training started............")
    gb_clf.fit(X_train,y_train)

    logger.debug("################ Best parameters ##############")
    logger.debug(f"Best Params: {gb_clf.best_params_}")
    
    return gb_clf



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    
    return y_pred


def get_confusion_matrix(y_actual, y_pred):
    """
    Create confusion matrix and save as image

    Args:
        y_actual (np.array): known labels
        y_pred (np.array): predicted labels
    
    Returns:
        cm : confusion matrix
    """
    
    cm = confusion_matrix(y_actual, y_pred)
    logger.debug(f"Confusion Matrix: {cm}")
    
    #confusion matrix to image
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title('Confusion Matrix')
    plt.xticks([0,1],['Predicted Negative', 'Predicted Positive'])
    plt.yticks([0,1], ['True Negative' , 'True Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    plt.savefig(os.path.join(IMAGE_PATH,'confusionmatrix.png'))
    plt.clf()
    
    return cm


def compute_slices(df, feature, y_true, y_pred):
    """
    Compute performance on slices for a given categorical feature
    a slice -->  value of a given feature is held fixed
    
    --------
    Args:
    
    df: 
        test dataframe, preprocessed, having features used for slices
    feature: 
        feature on which slicing is performed
    y_true: np.array
        true / known labels
    y_pred: np.array
        Predicted labels
        
    Return:
        dataframe
    """
    #get all slice options
    slices = df[feature].unique().tolist()
    
    df_perf = pd.DataFrame(index=slices, columns=['n_samples','precision','recall','fbeta'])
    
    for slice in slices:
        sliced_data = df[df[feature]==slice]
        #print(sliced_data)
        y_true_slice = y_true[df[feature]==slice]
        y_pred_slice = y_pred[df[feature]==slice]
        
        precision, recall, fbeta = compute_model_metrics(y_true_slice,y_pred_slice)
        df_perf.at[slice, 'feature'] = feature
        df_perf.at[slice, 'n_samples'] = len(y_true_slice)
        df_perf.at[slice, 'precision'] = precision
        df_perf.at[slice, 'recall'] = recall
        df_perf.at[slice, 'fbeta'] = fbeta
    
    return df_perf
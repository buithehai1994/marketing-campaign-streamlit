from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import streamlit as st
from tab_ml.logics import ML
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt

@st.cache_resource
def display_baseline_metrics(y_train):
    ml = ML()
    baseline_accuracy = ml.calculate_baseline_metrics(y_train)
    st.write(f"Baseline Accuracy: {baseline_accuracy}")

@st.cache_resource
def display_model_metrics(X_train, y_train, X_val, y_val, X_test, y_test, _model, average='weighted'):
    """
    Display the evaluation metrics for training, validation, and testing sets in a table format.

    Parameters:
    - x_train, x_val, x_test: Input features for training, validation, and testing sets
    - y_train, y_val, y_test: Target labels for training, validation, and testing sets
    - model: Trained machine learning model
    - average: Averaging strategy for precision, recall, and F1 score
    """
    # Compute metrics for training set
    train_pred = _model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(y_train, train_pred, average=average)

    # Compute metrics for validation set
    val_pred = _model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average=average)

    # Compute metrics for testing set
    test_pred = _model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, test_pred, average=average)

    # Prepare data for the table
    metrics_data = {
        'Set': ['Training', 'Validation', 'Testing'],
        'Accuracy': [train_accuracy, val_accuracy, test_accuracy],
        f'Precision ({average})': [train_precision, val_precision, test_precision],
        f'Recall ({average})': [train_recall, val_recall, test_recall],
        f'F1 Score ({average})': [train_f1, val_f1, test_f1]
    }

    # Create a DataFrame from the metrics data
    metrics_df = pd.DataFrame(metrics_data)

    # Display the table
    st.write("Evaluation Metrics:")
    st.table(metrics_df)
    
@st.cache_resource
def display_confusion_matrix(y_true, y_pred, class_labels=['Not subscribe', 'subscribe'], figsize=(8, 6)):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display confusion matrix using seaborn heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted: ' + label for label in class_labels],
                yticklabels=['Actual: ' + label for label in class_labels])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

@st.cache_resource
def metric(model,X_train,X_test,X_val,y_train,y_test,y_val):
    # Load model
    # ml=ML(trained_model=model)
    ml = ML()
    ml.load_model(model)
    metrics_train = ml.calculate_model_metrics(X_train, y_train)
    metrics_val = ml.calculate_model_metrics(X_val, y_val)
    metrics_test = ml.calculate_model_metrics(X_test, y_test)

    metrics_accuracy_training=ml.calculate_accuracy(X_train,y_train)
    metrics_accuracy_testing=ml.calculate_accuracy(X_test,y_test)
    metrics_accuracy_validation=ml.calculate_accuracy(X_val,y_val)

    # Create a DataFrame with all metrics
    metrics_data = {
        'Dataset': ['Training', 'Validation', 'Testing'],
        'Accuracy': [metrics_accuracy_training, metrics_accuracy_validation, metrics_accuracy_testing],
        'Precision': [metrics_train[1][0], metrics_val[1][0], metrics_test[1][0]],
        'Recall': [metrics_train[1][1], metrics_val[1][1], metrics_test[1][1]],
        'F1 Score': [metrics_train[1][2], metrics_val[1][2], metrics_test[1][2]]
        }
    # Display all metrics in one table
    st.write("Metrics for Training, Validation, and Testing")
    st.table(pd.DataFrame(metrics_data))

@st.cache_resource
def display_roc_curve(y_true, y_scores, ml_instance, title):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate AUC score
    auc_score = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guessing')
    plt.title(f'{title} - {ml_instance}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
@st.cache_resource
def display_correlation_matrix(X):
    """
    Plot the correlation matrix heatmap for the input features.

    Parameters:
    - X: Input features (DataFrame or array-like)
    """
    # Calculate correlation matrix
    ml=ML()
    corr_matrix = ml.calculate_correlation_matrix(X)
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title("Correlation Matrix")

    # Display the plot in Streamlit
    st.pyplot(fig)

# def display_metrics_and_visualizations(model, X_train, X_val, X_test, y_train, y_val, y_test):
    
#     metric(model,X_train=X_train,X_test=X_test,X_val=X_val,
#            y_train=y_train,y_test=y_test,y_val=y_val)

#     ml=ML()
#     ml.load_model(model)
#     model=ml.trained_model
#      # Make predictions
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
#     y_val_pred = model.predict(X_val)

#     # Display confusion matrix
#     st.header("Training Confusion Matrix")
#     display_confusion_matrix(y_train,y_train_pred,figsize=(6, 4))
#     # Add ROC curve display
#     st.header("ROC Curve for Training")
#     y_train_scores = model.predict_proba(X_train)[:, 1]
#     display_roc_curve(y_train, y_train_scores, ml_instance=model, figsize=(6, 4))
    
#     # Display confusion matrix
#     st.header("Validation Confusion Matrix")
#     display_confusion_matrix(y_val,y_val_pred,figsize=(6, 4))
#     # Add ROC curve display
#     st.header("ROC Curve for Validation")
#     y_val_scores = model.predict_proba(X_val)[:, 1]
#     display_roc_curve(y_val, y_val_scores, ml_instance=model, figsize=(6, 4))
    
#     # Display confusion matrix
#     st.header("Testing Confusion Matrix")
#     display_confusion_matrix(y_test,y_test_pred,figsize=(6, 4))  
#     # Add ROC curve display
#     st.header("ROC Curve for Testing")
#     y_test_scores = model.predict_proba(X_test)[:, 1]
#     display_roc_curve(y_test, y_test_scores, ml_instance=model, figsize=(6, 4))

@st.cache_resource
def display_model_performance_analysis():
    explanation_text = """
    **Model Performance Analysis**
    
    **Logistic Regression:**
    The Logistic Regression model shows consistent performance across datasets, but its scores are relatively lower compared to other models. While it does not exhibit signs of overfitting, its performance might not be as competitive as other models.
    
    **K-Nearest Neighbors (KNN):**
    The KNN models exhibit varying performances based on the number of neighbors. The model with 200 neighbors seems to generalize well but does not consistently outperform other models. This variation suggests sensitivity to hyperparameter tuning.
    
    **Random Forest:**
    Among the evaluated models, the Random Forest stands out for its consistent and high performance. The Best Forest with the specific parameters 'max_depth': 12.57, 'min_samples_leaf': 1.0, 'min_samples_split': 8.37, 'n_estimators': 144.93 achieves notable scores across all metrics on training, validation, and testing datasets. This indicates both excellent predictive ability and generalization.
    
    **Decision Tree:**
    The Decision Tree models also show competitive performance. The best-performing Decision Tree has parameters 'max_depth': 11.83, 'min_samples_leaf': 4.39, 'min_samples_split': 12.40. This tree achieves high scores across metrics on all datasets, indicating good generalization.
    
    **Support Vector Machine (SVM):**
    The SVM models perform reasonably well, and the best-performing SVM has parameters 'C': 8.35, 'gamma': 0.35. However, SVM's performance is slightly below that of the Random Forest and Decision Tree models.
    
    **Conclusion:**
    After careful analysis, the Random Forest model with the specific parameters (Best Forest ('max_depth': 12.57, 'min_samples_leaf': 1.0, 'min_samples_split': 8.37, 'n_estimators': 144.93))  appears to be the most robust and consistent performer among the evaluated models. It achieves high scores on all relevant metrics across training, validation, and testing datasets, indicating strong predictive power and generalization. This model is recommended for its balanced performance without clear signs of overfitting.

    However, despite the model's high scores in accuracy, precision, recall, and F1 score, there's a significant discrepancy in the ROC score between the training set (0.92) and the testing/validation sets (0.76). This discrepancy suggests that while the model performs well in classification metrics like accuracy, precision, and recall, it might struggle to clearly distinguish between the two classes when dealing with unseen data.
    """
    
    st.markdown(explanation_text)

@st.cache_resource
def feature_importance_explanation():
    explanation_text="""
    Feature importance refers to the measure of the impact or relevance of input variables (features) in a predictive model regarding its ability to predict the target outcome. In the context of machine learning algorithms like Random Forest, feature importance helps identify which features have the most significant influence on the model's predictions.
    Random Forest calculates feature importance based on how much each feature decreases the model's accuracy when it's not available for making predictions. This is often done by measuring the decrease in a criterion like Gini impurity or information gain when a particular feature is not included or is shuffled.
    The Random Forest feature importance analysis highlights the crucial role of the "euribor3m" variable in predicting the likelihood of campaign subscriptions. With a notably high feature importance score, "euribor3m" emerges as a key factor influencing customer behavior, aligning seamlessly with earlier findings in the Exploratory Data Analysis (EDA). The prominence of "euribor3m" in predicting campaign outcomes is attributed to its close connection with default rates and loans. 
    This insight underscores the financial implications of interest rates, as higher rates may burden individuals financially, thereby impacting their decision to subscribe to the campaign. The recommendation to monitor economic factors, especially interest rates, is a strategic response to this observation. By staying attuned to such economic indicators, marketing strategies can be dynamically adjusted to account for the potential impact of financial conditions on customer responsiveness. In essence, the Random Forest analysis not only reinforces the significance of "euribor3m" but also provides actionable insights for refining marketing strategies in the context of prevailing economic dynamics.
    """
    st.markdown(explanation_text)

@st.cache_resource
def display_cross_validation_analysis():
    explanation_text="""
    These results indicate that, compared to the baseline accuracy of 0.5, all models perform significantly better. This is a positive outcome and suggests that these models have learned patterns within the data and perform better than a random guest. 
    The Decision Tree model has the highest mean accuracy (0.8437), closely followed by the Random Forest model (0.8316). 
    The k-NN model also performs well with a mean accuracy of 0.7855. The Logistic Regression and SVM models have lower mean accuracies but still perform better than the baseline.These results indicate that, compared to the baseline accuracy of 0.5, all models perform significantly better. The Decision Tree model has the highest mean accuracy (0.8437), closely followed by the Random Forest model (0.8316). The k-NN model also performs well with a mean accuracy of 0.7855. 
    The Logistic Regression and SVM models have lower mean accuracies compared to the other algorithms.
    """

import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,precision_recall_fscore_support,roc_curve, auc
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class ML:
    def __init__(self):
        self.trained_model = None
        self.smote = SMOTE(random_state=42)
        self.scaler = StandardScaler()

    def split_data(self, X, y):
        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def oversample_data(self, X_train, y_train):
        # Apply SMOTE to balance the class distribution in the training set
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled
    
    def scale_data(self, X_train,X_test,X_val):
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)
        X_val_scaled  = self.scaler.transform(X_val)
        return X_train_scaled,X_test_scaled,X_val_scaled
        
    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.trained_model = pickle.load(file)

    def calculate_model_metrics(self, X, y):

         # Ensure X is a DataFrame and y is a Series
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert X to a NumPy array
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()

        if self.trained_model is None:
            raise ValueError("Model not loaded. Please load the model first.")

        # Use the loaded model for predictions and evaluation
        predictions = self.trained_model.predict(X)

        scores = precision_recall_fscore_support(y, predictions, average='weighted')

        return "precision, recall, F1 score:", scores

    def calculate_baseline_metrics(self, y):
        # Convert y to a Pandas Series if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()

        # Calculate the majority class in y
        y_mode = y.mode()[0] if len(y.mode()) > 0 else None

        # Create an array of ones with the same shape as y
        y_base = pd.Series(np.full(y.shape, y_mode), index=y.index)

        # Calculate metrics of the baseline model
        baseline_accuracy = accuracy_score(y, y_base)

        return baseline_accuracy

    def calculate_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def calculate_accuracy(self, X,y_true):
        
        # Use the loaded model for predictions and evaluation
        predictions = self.trained_model.predict(X)
        return accuracy_score(y_true, predictions)

    def calculate_roc_curve(self, X, y):
        if hasattr(self.trained_model, "decision_function"):
            y_scores = self.trained_model.decision_function(X)
        else:
            # Handle cases where decision_function is not available
            raise AttributeError("Model does not have decision_function method.")
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
        

    
            

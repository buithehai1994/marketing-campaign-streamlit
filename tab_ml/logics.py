import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class ML:
    def __init__(self):
        self.trained_model = None
        self.smote = SMOTE(random_state=42)
        self.scaler = StandardScaler()

    def split_data(self, X, y):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def coef(self):
        if self.trained_model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.trained_model.coef_

    def intercept(self):
        if self.trained_model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.trained_model.intercept_
        
    def oversample_data(self, X_train, y_train):
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled
    
    def scale_data(self, X_train, X_test, X_val):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled
        
    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.trained_model = pickle.load(file)
        
    def calculate_model_metrics(self, X, y):
        if self.trained_model is None:
            raise ValueError("Model not loaded. Please load the model first.")

        predictions = self.trained_model.predict(X)

        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='weighted')

        return precision, recall, f1

    def calculate_baseline_metrics(self, y):
        y_mode = y.mode()[0] if not y.mode().empty else None
        y_base = pd.Series(np.full(y.shape, y_mode), index=y.index)
        baseline_accuracy = accuracy_score(y, y_base)
        return baseline_accuracy

    def calculate_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def calculate_accuracy(self, X, y_true):
        if self.trained_model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        predictions = self.trained_model.predict(X)
        return accuracy_score(y_true, predictions)

    def calculate_roc_curve(self, X, y):
        if hasattr(self.trained_model, "decision_function"):
            y_scores = self.trained_model.decision_function(X)
            fpr, tpr, _ = roc_curve(y, y_scores)
            roc_auc = auc(fpr, tpr)
            return fpr, tpr, roc_auc
        else:
            raise AttributeError("Model does not have decision_function method.")

    def train_model(self, model, X_train, y_train):
        # Train the given model on the training data
        model.fit(X_train, y_train)
        # Set the trained model to the class attribute
        self.trained_model = model
        return model

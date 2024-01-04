import streamlit as st
import pandas as pd
import sys
import os
from imblearn.over_sampling import SMOTE
from pathlib import Path
import numpy as np

# Set Python path
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parents[0])
sys.path.append(parent_dir)

from tab_df.logics import Dataset
from tab_eda.logics import EDA
from tab_df.display import display_tab_df_content
# from tab_eda.display import display_tab_eda_report
from tab_eda.display import display_summary_statistics
from tab_eda.display import display_info
from tab_eda.display import display_missing_values
from tab_eda.display import display_plots,display_correlation_heatmap
from tab_encoding.display import display_tab_df_encoding_explain, display_correlation_encoding_heatmap
from tab_encoding.logics import Encoding
from tab_ml.display import display_baseline_metrics,display_model_metrics,display_confusion_matrix,metric, display_roc_curve, display_metrics_and_visualizations
from tab_ml.logics import ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import pickle
import csv

import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    # Suppress warnings related to feature names in Logistic Regression
    warnings.simplefilter("ignore", category=ConvergenceWarning)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CSV Explorer",
    page_icon=None,
    layout="wide",
)

# Display Title
st.title("Bank Data")

# Sidebar navigation for different sections

selected_tab = st.sidebar.radio("Navigation", ["Data", "EDA","Encoding", "Machine Learning Model"])

# Load data from "Data" tab
# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# set dataset
@st.cache_data
def fetch_data():
    # Specify the path to the CSV file relative to the app directory
    dataset_path = Path(__file__).resolve().parent.parent / "csv" / "TeleCom_Data.csv"
    dataset = Dataset()
    dataset.set_data(dataset_path)
    return dataset

dataset = fetch_data()
data_from_tab_df = pd.DataFrame(dataset.data)
eda = EDA(data_from_tab_df)

# Metrics for training, validation, and testing


# Display content based on selected sidebar tab
if selected_tab == "Data":
    st.sidebar.header("Data")
    display_tab_df_content(dataset)
elif selected_tab == "EDA":
    st.sidebar.header("EDA")
 
    # Create sub-tabs for EDA section
    tab_titles = ["Summary Statistics", "Plots"]
    selected_sub_tab = st.sidebar.radio("Sub-navigation",tab_titles)


    if selected_sub_tab == tab_titles[0]:
        st.header(f"Summary Statistics")
        # Create sub-sub-tabs for Correlation
        sub_tab_titles=["Summary", "Info", "Missing Values"]
        selected_sub_sub_tab = st.sidebar.radio("Dataset", sub_tab_titles)

        if selected_sub_sub_tab ==sub_tab_titles[0]:
            display_summary_statistics(data_from_tab_df)
        
        elif selected_sub_sub_tab== sub_tab_titles[1]:
            display_info(data_from_tab_df)
        
        else:
            display_missing_values(data_from_tab_df)

        # Display summary statistics
        # Replace this with the function to display summary statistics

    else:
        st.header(f"Plots for data")
        display_plots(data_from_tab_df)

elif selected_tab == "Encoding":
    encoding=Encoding(data=data_from_tab_df)
    data_for_ml=encoding.label_encoding()
    display_tab_df_encoding_explain(data_for_ml)

else:  # "Machine Learning Model" tab
    pass
    st.sidebar.header("Machine Learning Model")
    # Placeholder for ML content
    st.sidebar.write("This tab can contain content related to your machine learning model.")
    # Create sub-tabs for EDA section
    tab_titles = ['Base line model and Cross Validation','LogisticRegression','KNN','RandomForest',
                  'DecisionTree','SVM']

    selected_sub_tab = st.sidebar.radio("Sub-navigation",tab_titles)


    
    @st.cache_data
    def load_data(file_path):
        return pd.read_csv(file_path, header=None, skiprows=1)

    # Define file paths
    y_train_path = 'csv/y_train.csv'
    y_test_path = 'csv/y_test.csv'
    y_val_path = 'csv/y_val.csv'
    X_train_path = 'csv/X_train.csv'
    X_test_path = 'csv/X_test.csv'
    X_val_path = 'csv/X_val.csv'

    # Load data using the cache function
    y_train = load_data(y_train_path)
    y_test = load_data(y_test_path)
    y_val = load_data(y_val_path)
    X_train = load_data(X_train_path)
    X_test = load_data(X_test_path)
    X_val = load_data(X_val_path)

    if selected_sub_tab==tab_titles[0]:
        display_baseline_metrics(y_train)
        cross_validation_table = pd.read_csv("csv/cross_validation_results.csv")
        st.write("Cross validation results")
        st.table(cross_validation_table)

    if selected_sub_tab==tab_titles[1]:
        # Create sub-tabs
        selected_sub_sub_tab = st.sidebar.radio("Sub-navigation",["Default params", "Regularization"])

        if selected_sub_sub_tab=="Default params":
            # Load model
            selected_model='app/log_reg.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Regularization":
            # Load model
            selected_model='app/log_elastic_reg.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
    if selected_sub_tab==tab_titles[2]:
         # Create sub-tabs
        selected_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                ["KNN (n_neighbors=15 and metric: ‘Euclidean')", 
                                                 "KNN (n_neighbors=55 and metric: ‘Euclidean')",
                                                 "KNN (n_neighbors=100 and metric: ‘Euclidean')",
                                                 "KNN (n_neighbors=200 and metric: ‘Euclidean')"
                                                 ])
        if selected_sub_sub_tab=="KNN (n_neighbors=15 and metric: ‘Euclidean')":
            selected_model='app/knn_15_euc.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="KNN (n_neighbors=55 and metric: ‘Euclidean')":
            selected_model='app/knn_55_euc.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="KNN (n_neighbors=100 and metric: ‘Euclidean')":
            selected_model='app/knn_100_euc.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="KNN (n_neighbors=200 and metric: ‘Euclidean')":
            selected_model='app/knn_200_euc.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
    if selected_sub_tab==tab_titles[3]:
         # Create sub-tabs
        selected_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                [
                                                "rf6 (random_state=8, n_estimators=50, max_depth=5)", 
                                                 "rf8 (random_state=8, n_estimators=50, max_depth=15, min_samples_leaf=10)",
                                                 "rf11 (random_state=8, n_estimators=50, max_depth=15, min_samples_leaf=2, max_features=5)",
                                                 "Best Forest ('max_depth': 12.57, 'min_samples_leaf': 1.0, 'min_samples_split': 8.37, 'n_estimators': 144.93)"
                                                 ])
        if selected_sub_sub_tab=="rf6 (random_state=8, n_estimators=50, max_depth=5)":
            selected_model='app/rf6.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="rf8 (random_state=8, n_estimators=50, max_depth=15, min_samples_leaf=10)":
            selected_model='app/rf8.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="rf11 (random_state=8, n_estimators=50, max_depth=15, min_samples_leaf=2, max_features=5)":
            selected_model='app/rf11.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Best Forest ('max_depth': 12.57, 'min_samples_leaf': 1.0, 'min_samples_split': 8.37, 'n_estimators': 144.93)":
            selected_model='app/best_forest.pkl'

            selected_sub_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                ["Optimization Results",
                                                 "Model Performance"
                                                 ])
            if selected_sub_sub_sub_tab=="Optimization Results":
            
                # Display the content of "best_forest_optimization_results.csv"
                best_forest_results = pd.read_csv("rf_optimization_results.csv")

                # Find the row with the maximum value in the "target" column
                max_row_index = best_forest_results['target'].idxmax()

                # Highlight the row with the maximum value
                highlighted_results = best_forest_results.style.apply(lambda x: 
                ['background: lightgreen' if x.name == max_row_index else '' for _ in x], axis=1)
                
                # Display the DataFrame with highlighted row in Streamlit
                st.header("Random Forest Optimization Results")
                st.table(highlighted_results.hide(axis="index"))
            
            if selected_sub_sub_sub_tab=="Model Performance":
                display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                                y_train=y_train, y_val=y_val, y_test=y_test)

    if selected_sub_tab==tab_titles[4]:
        # Create sub-tabs
        selected_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                [
                                                "Tree 1(random_state=42)",
                                                "Tree 2(random_state=42, min_samples_split=5)",
                                                "Tree 3(random_state=42, min_samples_split=20)",
                                                "Tree 4(random_state=42, min_samples_split=5, max_depth=3)",
                                                "Tree 5(random_state=42, min_samples_split=5, max_depth=4)",
                                                "Best Tree ('max_depth': 11.83, 'min_samples_leaf': 4.39, 'min_samples_split': 12.40)"                                    
                                                    ])
        if selected_sub_sub_tab=="Tree 1(random_state=42)":
            selected_model='app/tree_1.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Tree 2(random_state=42, min_samples_split=5)":
            selected_model='app/tree_2.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Tree 3(random_state=42, min_samples_split=20)":
            selected_model='app/tree_3.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Tree 4(random_state=42, min_samples_split=5, max_depth=3)":
            selected_model='app/tree_4.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Tree 5(random_state=42, min_samples_split=5, max_depth=4)":
            selected_model='app/tree_5.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Best Tree ('max_depth': 11.83, 'min_samples_leaf': 4.39, 'min_samples_split': 12.40)":
            selected_model='app/best_tree.pkl'

            selected_sub_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                ["Optimization Results",
                                                 "Model Performance"
                                                 ])
            if selected_sub_sub_sub_tab=="Optimization Results":
            
                # Display the content of "decision_tree_optimization_results"
                best_tree_results = pd.read_csv("decision_tree_optimization_results")

                # Find the row with the maximum value in the "target" column
                max_row_index = best_tree_results['target'].idxmax()

                # Highlight the row with the maximum value
                highlighted_results = best_tree_results.style.apply(lambda x: 
                ['background: lightgreen' if x.name == max_row_index else '' for _ in x], axis=1)
                
                # Display the DataFrame with highlighted row in Streamlit
                st.header("Decision Tree Optimization Results")
                st.table(highlighted_results.hide(axis="index"))
            
            if selected_sub_sub_sub_tab=="Model Performance":
                display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                                y_train=y_train, y_val=y_val, y_test=y_test)
                
    if selected_sub_tab==tab_titles[5]:
        # Create sub-tabs
        selected_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                [
                                                "SVC_1 (default parameters)",
                                                "SVC_2 (C=0.5)",
                                                "SVC_3 (C=1)",
                                                "SVC_4 (C=1.5)",
                                                "Best SVC ('C': 8.35, 'gamma': 0.35)"                                    
                                                    ])
        if selected_sub_sub_tab=="SVC_1 (default parameters)":
            selected_model='app/svc_1.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="SVC_2 (C=0.5)":
            selected_model='app/svc_2.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="SVC_3 (C=1)":
            selected_model='app/svc_3.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="SVC_4 (C=1.5)":
            selected_model='app/svc_4.pkl'
            display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
            
        if selected_sub_sub_tab=="Best SVC ('C': 8.35, 'gamma': 0.35)":
            selected_model='app/best_svc.pkl'

            selected_sub_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                ["Optimization Results",
                                                 "Model Performance"
                                                 ])
            if selected_sub_sub_sub_tab=="Optimization Results":
                
                @st.cache_data
                def fetch_best_svc_results():
                    data=pd.read_csv("app/svm_optimization_results.csv")
                    return data
                # Display the content of "svm_optimization_results"
                best_svc_results = fetch_best_svc_results()

                # Find the row with the maximum value in the "target" column
                max_row_index = best_svc_results['target'].idxmax()

                # Highlight the row with the maximum value
                highlighted_results = best_svc_results.style.apply(lambda x: 
                ['background: lightgreen' if x.name == max_row_index else '' for _ in x], axis=1)
                
                # Display the DataFrame with highlighted row in Streamlit
                st.header("SVM Optimization Results")
                st.table(highlighted_results.hide(axis="index"))
            
            if selected_sub_sub_sub_tab=="Model Performance":
                display_metrics_and_visualizations(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                                y_train=y_train, y_val=y_val, y_test=y_test)

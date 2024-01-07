import streamlit as st
import pandas as pd
import sys
import os
from imblearn.over_sampling import SMOTE
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import openpyxl
import base64
import imblearn
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import dataprep
import re

# Set Python path
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parents[0])
sys.path.append(parent_dir)

from tab_df.logics import Dataset
from tab_eda.logics import EDA
from tab_df.display import display_tab_df_content
# from tab_eda.display import display_tab_eda_report
from tab_eda.display import display_missing_values,display_plots,display_correlation_heatmap,display_analysis,display_info,display_summary_statistics,display_stack_bar_chart, display_plot_distribution, display_generate_visual_eda_report
from tab_encoding.display import display_tab_df_encoding_explain, display_correlation_encoding_heatmap
from tab_encoding.logics import Encoding
from tab_ml.display import display_baseline_metrics,display_model_metrics,display_confusion_matrix,metric, display_roc_curve, display_metrics_and_visualizations,display_model_performance_analysis,display_cross_validation_analysis, feature_importance_explanation
from tab_ml.logics import ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tab_intro.introduction import display_introduction
from dataprep.eda import plot,create_report
import pickle
import csv
import streamlit.components.v1 as components
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

selected_tab = st.sidebar.radio("Navigation", ["Introduction", "Data", "EDA","Encoding", "Machine Learning Model","Feature Importance","Deployment","Ethical Consideration", "References"])

# Load data from "Data" tab
# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Function to generate and display word cloud for a given text
def display_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()
    
def remove_hyperlinks(html_content):
    # Remove anchor tags (<a>) from the HTML content
    return re.sub(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>', r'<span>\1</span>', html_content)

# set dataset
# @st.cache_data
def fetch_data():
    # Specify the path to the CSV file relative to the app directory
    dataset_path = Path(__file__).resolve().parent.parent / "csv" / "TeleCom_Data.csv"
    dataset = Dataset()
    dataset.set_data(dataset_path)
    return dataset

dataset = fetch_data()

data_from_tab_df = pd.DataFrame(dataset.data)

eda = EDA(data_from_tab_df)

# @st.cache_data()
def generate_download_link(df):
    csv_data = df.to_csv(index=False).encode()
    b64_csv = base64.b64encode(csv_data).decode()
    return f'<a href="data:text/csv;charset=utf-8;base64,{b64_csv}" download="TeleCom_Data.csv">Download CSV File</a>'

def perform_encoding():
    encoding = Encoding(data=data_from_tab_df)
    data_for_ml = encoding.label_encoding()
    return data_for_ml
    
data_for_ml = perform_encoding()

# Function to read and display the HTML file using streamlit_elements

# @st.cache
def read_html_content(saved_html_file_path):
    with open(saved_html_file_path, 'r',encoding='utf-8') as file:
        html_code = file.read()
    return html_code
    
    
# def eda_report(data_from_tab_df):
#     st.title("Exploratory Data Analysis Report")
    
#     # Display EDA report using dataprep
#     plot(data_from_tab_df)
#     st.pyplot()  # Display the generated plots using Streamlit
    
# Instantiate the ML class
ml_instance = ML()

# Extract features and target variable
X = data_for_ml.drop(['y','poutcome','contact','default','previous','emp.var.rate','month',
                      'cons.price.idx','job','age','cons.conf.idx','campaign','duration','marital'], axis=1)
y = data_for_ml['y']

from sklearn.model_selection import train_test_split

# Split the data into training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = ml_instance.split_data(X, y)

# Oversample the training data
from imblearn.over_sampling import SMOTE
X_train, y_train = ml_instance.oversample_data(X_train, y_train)

# scale the data
from sklearn.preprocessing import StandardScaler
# Scale the training data
X_train,X_test,X_val  = ml_instance.scale_data(X_train,X_test,X_val)

  # Display content based on selected sidebar tab
if selected_tab =="Introduction":
    display_introduction()
elif selected_tab == "Data":
    st.sidebar.header("Data")
    display_tab_df_content(dataset)
    st.write("The dataset exhibits problems like absent data and inaccuracies. The subsequent step involves conducting EDA to scrutinize and understand the data thoroughly.")
    # Display the download link
    link=generate_download_link(data_from_tab_df)
    st.markdown(link, unsafe_allow_html=True)
elif selected_tab == "EDA":
    st.sidebar.header("EDA")
 
    # Create sub-tabs for EDA section
    tab_titles = ["Summary Statistics", "Plots","Analyse Relationship Between Variables"]
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
            
    if selected_sub_tab == tab_titles[1]:
        eda_report_path=Path(__file__).resolve().parent.parent / "app" / "report.html"
        
        #   # Read HTML content from the file
        # with open(eda_report_path, 'r') as file:
        #     html_content = file.read()
        # # Remove hyperlinks from the HTML content
        # html_content_no_links = remove_hyperlinks(html_content)
        
        # # Display modified HTML content in the Streamlit app
        # components.html(html_content_no_links, width=800, height=600)

        # report = create_report(data_from_tab_df, title='EDA Report')
        
        # # Get the HTML content from the Container object
        # html_content = report.show_browser()

        # # Render the HTML content within Streamlit
        # st.components.v1.html(eda_html, width=800, height=600, scrolling=True)
        # Render the HTML content within an iframe in Streamlit
        # Display the HTML report in Streamlit using st.markdown
        # st.markdown(html_content, unsafe_allow_html=True)
        # Display the HTML report in Streamlit using an iframe
        st.markdown(
            f'<iframe src="data:text/html;base64,{eda_report_path}" width=900 height=600></iframe>',
            unsafe_allow_html=True
        )

    if selected_sub_tab == tab_titles[2]:
        sub_tab_titles = ["Graph","Analysis"]
        selected_sub_sub_tab = st.sidebar.radio("Sub-navigation",sub_tab_titles)        
        if selected_sub_sub_tab == "Graph":
            # Plot the graph displaying relationship between interest rate and other variables
            display_stack_bar_chart(df=data_from_tab_df)            
        if selected_sub_sub_tab == "Analysis":
            image_path = (Path(__file__).resolve().parent.parent / "tab_eda" / "business cycle.png").as_posix()
            display_analysis(image_path=image_path)               
elif selected_tab == "Encoding":
    display_tab_df_encoding_explain(data_for_ml)

elif selected_tab == "Machine Learning Model":
    pass
    st.sidebar.header("Machine Learning Model")
    # Placeholder for ML content
    st.sidebar.write("This tab can contain content related to your machine learning model.")
    # Create sub-tabs for EDA section
    tab_titles = ['Base line model and Cross Validation','LogisticRegression','KNN','RandomForest',
                  'DecisionTree','SVM','Model evaluaton']

    selected_sub_tab = st.sidebar.radio("Sub-navigation",tab_titles)

    
    @st.cache_data
    def get_model_metrics(model, X_train, X_val, X_test, y_train, y_val, y_test):
        return display_metrics_and_visualizations(model, X_train, X_val, X_test, y_train, y_val, y_test)

    if selected_sub_tab==tab_titles[0]:
        display_baseline_metrics(y_train)
        cross_validation_table = pd.read_csv("csv/cross_validation_results.csv")
        st.write("Cross validation results")
        st.table(cross_validation_table)
        display_cross_validation_analysis()
    if selected_sub_tab==tab_titles[1]:
        # Create sub-tabs
        selected_sub_sub_tab = st.sidebar.radio("Sub-navigation",["Default params", "Regularization"])

        if selected_sub_sub_tab=="Default params":
            # Load model
            selected_model='app/log_reg.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Regularization":
            # Load model
            selected_model='app/log_elastic_reg.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
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
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="KNN (n_neighbors=55 and metric: ‘Euclidean')":
            selected_model='app/knn_55_euc.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="KNN (n_neighbors=100 and metric: ‘Euclidean')":
            selected_model='app/knn_100_euc.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="KNN (n_neighbors=200 and metric: ‘Euclidean')":
            selected_model='app/knn_200_euc.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
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
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="rf8 (random_state=8, n_estimators=50, max_depth=15, min_samples_leaf=10)":
            selected_model='app/rf8.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="rf11 (random_state=8, n_estimators=50, max_depth=15, min_samples_leaf=2, max_features=5)":
            selected_model='app/rf11.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Best Forest ('max_depth': 12.57, 'min_samples_leaf': 1.0, 'min_samples_split': 8.37, 'n_estimators': 144.93)":
            selected_model='app/best_forest.pkl'

            selected_sub_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                ["Optimization Results",
                                                 "Model Performance"
                                                 ])
            if selected_sub_sub_sub_tab=="Optimization Results":
            
                # Display the content of "best_forest_optimization_results.csv"
                best_forest_results = pd.read_csv("csv/rf_optimization_results.csv")

                # Find the row with the maximum value in the "target" column
                max_row_index = best_forest_results['target'].idxmax()

                # Highlight the row with the maximum value
                highlighted_results = best_forest_results.style.apply(lambda x: 
                ['background: lightgreen' if x.name == max_row_index else '' for _ in x], axis=1)
                
                # Display the DataFrame with highlighted row in Streamlit
                st.header("Random Forest Optimization Results")
                st.table(highlighted_results.hide(axis="index"))
            
            if selected_sub_sub_sub_tab=="Model Performance":
                get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
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
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Tree 2(random_state=42, min_samples_split=5)":
            selected_model='app/tree_2.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Tree 3(random_state=42, min_samples_split=20)":
            selected_model='app/tree_3.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Tree 4(random_state=42, min_samples_split=5, max_depth=3)":
            selected_model='app/tree_4.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Tree 5(random_state=42, min_samples_split=5, max_depth=4)":
            selected_model='app/tree_5.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="Best Tree ('max_depth': 11.83, 'min_samples_leaf': 4.39, 'min_samples_split': 12.40)":
            selected_model='app/best_tree.pkl'

            selected_sub_sub_sub_tab = st.sidebar.radio("Sub-navigation",
                                                ["Optimization Results",
                                                 "Model Performance"
                                                 ])
            if selected_sub_sub_sub_tab=="Optimization Results":
            
                # Display the content of "decision_tree_optimization_results"
                best_tree_results = pd.read_csv("csv/decision_tree_optimization_results.csv")

                # Find the row with the maximum value in the "target" column
                max_row_index = best_tree_results['target'].idxmax()

                # Highlight the row with the maximum value
                highlighted_results = best_tree_results.style.apply(lambda x: 
                ['background: lightgreen' if x.name == max_row_index else '' for _ in x], axis=1)
                
                # Display the DataFrame with highlighted row in Streamlit
                st.header("Decision Tree Optimization Results")
                st.table(highlighted_results.hide(axis="index"))
            
            if selected_sub_sub_sub_tab=="Model Performance":
                get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
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
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="SVC_2 (C=0.5)":
            selected_model='app/svc_2.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="SVC_3 (C=1)":
            selected_model='app/svc_3.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                               y_train=y_train, y_val=y_val, y_test=y_test)
            
        if selected_sub_sub_tab=="SVC_4 (C=1.5)":
            selected_model='app/svc_4.pkl'
            get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
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
                    data=pd.read_csv("csv/svm_optimization_results.csv")
                    return data
                # Display the content of "csv/svm_optimization_results"
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
                get_model_metrics(model=selected_model, X_train=X_train, X_val=X_val, X_test=X_test, 
                                                y_train=y_train, y_val=y_val, y_test=y_test)

    if selected_sub_tab==tab_titles[6]:
         display_model_performance_analysis()
         
elif selected_tab == "Feature Importance":
    st.header("Feature Importance")
    rf_model="app/best_forest.pkl"
    # Load the trained model
    ml = ML()
    ml.load_model(rf_model)

    # Load the training data
    X_train_path = 'csv/X_train.csv'
    X_train = pd.read_csv(X_train_path, header=0)

    if ml.trained_model and X_train is not None:
        # Get feature names
        feature_names = list(X_train.columns)

        # Get feature importances
        feature_importances = ml.trained_model.feature_importances_

        # Create a DataFrame with feature names and their corresponding importances
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

        # Sort the features by importance in descending order
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # Display the sorted feature importances as a table
        st.write("Sorted Feature Importances:")
        st.table(feature_importance_df)

        # Create a bar plot to visualize feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        st.pyplot(plt)  # Show the Matplotlib plot in Streamlit
        feature_importance_explanation()
    else:
        st.write("Error: Model or data not found. Please check the paths and ensure they are correct.")

elif selected_tab == "Deployment":
    rf_model = "app/best_forest.pkl"
    # Load the trained model
    ml = ML()
    ml.load_model(rf_model)

    st.header("Model Deployment")

    X_train_path = 'csv/X_train.csv'
    X_train = pd.read_csv(X_train_path, header=0)
    feature_columns = list(X_train.columns)

    # Dictionary containing the categorical feature values
    categorical_features = {
        'education': {'unknown': 0, 'illiterate': 0, 'basic.4y': 0, 'basic.6y': 0, 'basic.9y': 1, 'high.school': 0,
                      'professional.course': 1, 'university.degree': 2},
        'housing': {'yes': 1, 'no': 0, 'unknown': 0},
        'loan': {'yes': 1, 'no': 0, 'unknown': 0},
        'y': {'yes': 1, 'no': 0},
        'day_of_week': {'thu': 5, 'mon': 2, 'wed': 4, 'tue': 3, 'fri': 6},
    }

    # Input fields for the user to enter new data for prediction
    st.subheader("Enter New Data for Prediction")
    new_data = {}
    for feature in feature_columns:
        if feature == 'euribor3m':
            # Slider for 'euribor3m' feature within the range of 0 to 7%
            value = st.slider(f"Select value for {feature} %", 0.2, 7.0, 0.2, step=0.001)
            # Store the selected value for 'euribor3m'
            new_data[feature] = value
        elif feature in categorical_features:
            # Dropdown for other categorical features
            value = st.selectbox(f"Select value for {feature}", list(categorical_features[feature].keys()))
            # Store the selected categorical value as it is
            new_data[feature] = value
        else:
            # Text input for continuous features
            new_data[feature] = st.text_input(f"Enter value for {feature}", "")

    # Make prediction when the user clicks the "Predict" button
    if st.button("Predict"):
        # Create a DataFrame with the user-input data
        input_data = pd.DataFrame([new_data])

        # Ensure the input DataFrame has the same columns as the training data
        missing_cols = set(feature_columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0  # Add missing columns with default values

        # Use the 'make_prediction' method from the ML class to get the prediction
        encoded_input_data = pd.DataFrame()
        for col, values in categorical_features.items():
            if col in input_data.columns:
                encoded_input_data[col] = input_data[col].map(values)

        # Include numerical features as they are
        numerical_columns = set(input_data.columns) - set(categorical_features.keys())
        for col in numerical_columns:
            if col in input_data.columns:
                encoded_input_data[col] = input_data[col]
        st.write("Encoded input data")
        st.write(encoded_input_data)

        # Scale the input data using the same scaler used in the ML class
        X_input_scaled = ml_instance.scaler.transform(encoded_input_data)

        # Use the 'ml' object to make predictions with the encoded input data
        prediction = ml.trained_model.predict(X_input_scaled)

        # Convert scaled data back to a DataFrame and assign column names
        X_input_scaled_df = pd.DataFrame(X_input_scaled, columns=encoded_input_data.columns)
        st.write("Scaled input data")
        st.write(X_input_scaled_df)
        
        if int(prediction)==0:
            display_word_cloud("Not Subscribe")
        elif int(prediction)==1:
            display_word_cloud("Subsribe")
        else:
            print("Error")
elif selected_tab == "Ethical Consideration":
    st.write("It is essential to handle the results of this experiment with care and responsibility, ensuring that the data used, which includes sensitive information about customers (such as age and marital), is not misused for purposes beyond the specified business objectives.")
    st.write("Additionally, regular monitoring of the machine learning model's performance is crucial. Continuous evaluation helps verify the model's accuracy with unseen data and aids in identifying any potential ethical or practical issues. This ongoing assessment ensures the model's recommendations remain ethical and reliable.")

elif selected_tab == "References":
    citation = "Sexton, R. L. (2010). *Exploring Economics* (5th ed.). Mason, OH: South Western Educational Publishing."
    st.write(citation)
    

import streamlit as st
from tab_encoding.logics import Encoding
from tab_eda.logics import EDA

def display_tab_df_encoding_explain(dataset):
    st.subheader("Encoded DataFrame:")
    
    explanation_text = """
    ### Encoding Categorical Columns and Transforming Numeric Data
    
    The dataset has undergone preprocessing steps that involve encoding categorical columns and transforming numerical data for better model performance:
    
    #### Categorical Column Encoding:
    
    - **'month' Encoding:** The 'month' column, representing different months of the year, has been converted into numerical values. Each month is encoded with a corresponding integer (e.g., 'jan' as 1, 'feb' as 2, 'mar' as 3, etc.).
    
    - **'education' Encoding:** Educational qualifications in the 'education' column have been categorized into numerical groups representing various education levels. These categories have been encoded into numeric values (e.g., 'high.school' as 1, 'university.degree' as 2, etc.).
    
    - **'job' Encoding:** Categorical job titles in the 'job' column have been transformed into numeric representations. Each job category is encoded using specific numeric values that correspond to different job types (e.g., 'admin.' as 1, 'blue-collar' as 2, etc.).
    
    #### Numeric Data Transformation:
    
    - **Log Transformation:** Certain numerical columns like 'age', 'duration', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', and 'campaign' have undergone a logarithmic transformation. This transformation helps in normalizing the data and reducing skewness, improving the performance of machine learning models.
    
    - **Outlier Removal:** The data has been cleaned by removing outliers in columns like 'duration', 'age', 'emp.var.rate', 'cons.conf.idx', 'cons.price.idx', and 'campaign'. This process ensures that extreme values, which could adversely impact model training, have been eliminated.
    
    - **Potential Error Handling:** The majority of values are 999, indicating that a significant portion did not have a previous contact or the exact timing is unknown, coded as 999. Therefore, it is recommended that we should not use pdays for analysis.
    
    These preprocessing steps enable the dataset to be in a more suitable format for machine learning algorithms, enhancing their ability to derive meaningful insights and make accurate predictions.
    """
    
    st.markdown(explanation_text)
    st.write(dataset.head())  # Display or return the encoded data frame


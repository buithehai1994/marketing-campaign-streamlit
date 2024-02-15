import streamlit as st
import pandas as pd
from pathlib import Path
import openpyxl

def display_introduction():

    explanation_text = """
    ### Overview of the Project
    The purpose of this assessment is to perform exploratory data analysis (EDA) on a marketing campaign dataset from a telecommunications company 
    and subsequently construct data science models to provide valuable insights into two critical business inquiries. 
    These inquiries involve identifying customer segments that are most responsive to marketing campaigns and formulating effective business strategies. 
    The telecommunication company recently initiated a marketing campaign to encourage customers to adopt their new subscription plan. 
    They are seeking assistance in gaining a comprehensive understanding of their customer base and pinpointing segments that exhibit the highest responsiveness to marketing 
    initiatives. The response variable, "y," indicates whether a client subscribed to the new plan, which was the primary objective of the campaign.
    """
    
    st.markdown(explanation_text)  # Display the explanatory text
    
    dataset_path = Path(__file__).resolve().parent.parent / "tab_intro" / "Data_dictionary.xlsx"

    dictionary=pd.read_excel(dataset_path)

    # Display a sample of the dataset
    st.write("Data Dictionary:")
    st.write(dictionary)


def display_univariate_introduction():

    explanation_text = """
    The goal of this experiment is to examine the relationship between wealth and cancer death rate (**TARGET_deathRate**). Therefore, I will choose two variables, namely **medIncome** and **povertyPercent** as independent variables and train two univariate linear regression models. 
    **medIncome** variable represents the median income per US county while **povertyPercent** calculates percent of the populace in poverty. As a result, these two independent variables should demonstrate the relationship between wealth and cancer rate.
    The results of this study may indicate a potential inequality in healthcare treatment between the rich and the poor. The costs of treatment or standard of living may be the reasons for this imparity. Despite the high fee of cancer treatment, the fee for insurance is more affordable. A reasonable price insurance package with an effective mechanism for people with low incomes can be the solution to shorten the gap in cancer diagnosis and treatment. As a result, micro insurance products, which offer coverage for poor people with little savings, should be promoted.
    """
    
    st.markdown(explanation_text)  # Display the explanatory text


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


import pandas as pd

class EDA:
    def __init__(self, data):
        self.data = data

    def perform_eda(file_path):
        df = pd.read_csv(file_path)
        
        # Display summary statistics
        st.subheader("Summary Statistics:")
        st.write(df.describe())

        # Display column info
        st.subheader("Column Info:")
        st.write(df.info())

        # Display missing values count
        st.subheader("Missing Values Count:")
        st.write(df.isnull().sum())
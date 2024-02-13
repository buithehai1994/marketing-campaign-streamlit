from sklearn.preprocessing import LabelEncoder
import numpy as np
import streamlit as st

class Encoding:
    def __init__(self, data):
        self.data = data

        # Check if data is available
        if self.data is None:
            raise ValueError("No data available. Use set_data() to load data first.")

    def label_encoding(self):
        # Perform label encoding and other preprocessing steps
        # Access columns using self.data.get_column('column_name')

        # Ensure to return the processed data
        return self.data

    @st.cache
    def splitting_x(self):
        # Split features
        return self.data.drop(['y','poutcome','contact','default','previous','emp.var.rate','month',
                               'cons.price.idx','job','age','cons.conf.idx','campaign','duration','marital'], axis=1)

    @st.cache
    def splitting_y(self):
        # Split target
        return self.data['y']

    @st.cache
    def head_df(self):
        if self.data is not None:
            return self.data.head()
        else:
            return "No data available"

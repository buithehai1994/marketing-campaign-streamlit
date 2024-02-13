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
        if self.data is None:
            raise ValueError("No data available. Use set_data() to load data first.")
        
        # Log transformation on 'duration' column
        self.data['duration'] = np.log(self.data['duration'] + 0.01)
        self.data = self.data[(self.data['duration'] < 7.42) & (self.data['duration'] > 3.045)]

        # Log transformation on 'age' column
        self.data['age'] = np.log(self.data['age'])
        self.data = self.data[(self.data['age'] > 2.89) & (self.data['age'] < 4.419)]

        # Mapping categorical variables
        self.data['month'] = self.data['month'].map({'may':5, 'jun':6, 'jul':7, 'aug':8, 'oct':10, 'nov':11, 'dec':12, 'mar':3, 'apr':4, 'sep':9}).astype(int)
        self.data['education'] = self.data['education'].map({'unknown':0,'illiterate':0,'basic.4y':0,'basic.6y':0, 'basic.9y':1,'high.school':0,
                                                             'professional.course':1, 'university.degree':2})
        self.data['job'] = self.data['job'].map({'admin.':1, 'blue-collar':2, 'technician':1,'services':2,'management':3, 'entrepreneur':3,'self-employed':3,'retired':3, 
                                                  'unemployed':3,'housemaid':3,'unknown':3,'student':3})
        self.data['marital'] = self.data['marital'].map({'married':0, 'single':1, 'divorced':1,'unknown':1})
        self.data['default'] = self.data['default'].map({'yes':1, 'no':0, 'unknown':1})
        self.data['housing'] = self.data['housing'].map({'yes' : 1, 'no': 0,'unknown':0})
        self.data['loan'] = self.data['loan'].map({'yes' : 1, 'no': 0,'unknown':0})
        self.data['y'] = self.data['y'].map({'yes' : 1, 'no': 0})
        self.data['contact'] = self.data['contact'].map({'cellular' : 1, 'telephone': 0})
        self.data['day_of_week'] = self.data['day_of_week'].map({'thu' : 5, 'mon': 2,'wed':4,'tue':3,'fri':6})
        self.data['poutcome'] = self.data['poutcome'].map({'nonexistent' : 0, 'failure': 1,'success':1})

        # Shift and log transformation on 'emp.var.rate' column
        min_value_emp_var_rate = self.data['emp.var.rate'].min()
        shift_constant_emp_var_rate = abs(min_value_emp_var_rate) + 0.1
        self.data['emp.var.rate'] = np.log(self.data['emp.var.rate'] + shift_constant_emp_var_rate)

        # Shift and log transformation on 'cons.conf.idx' column
        min_value_cons_conf_idx = self.data['cons.conf.idx'].min()
        shift_constant_cons_conf_idx = abs(min_value_cons_conf_idx) + 0.1
        self.data['cons.conf.idx'] = np.log(self.data['cons.conf.idx'] + shift_constant_cons_conf_idx)
        self.data = self.data[self.data['cons.conf.idx'] >= 1.569]

        # Log transformation on 'cons.price.idx', 'euribor3m', 'nr.employed', and 'campaign' columns
        self.data['cons.price.idx'] = np.log(self.data['cons.price.idx'])
        self.data['euribor3m'] = np.log(self.data['euribor3m'])
        self.data['nr.employed'] = np.log(self.data['nr.employed'])
        self.data['campaign'] = np.log(self.data['campaign'])
        self.data = self.data[self.data['campaign'] <= 2.708]

        # Drop columns 'pdays' and 'nr.employed'
        self.data = self.data.drop(['pdays','nr.employed'],axis=1)

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

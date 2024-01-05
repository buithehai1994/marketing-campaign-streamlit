import pandas as pd
from io import StringIO
from tab_df.logics import Dataset
# from tabulate import tabulate
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from dataprep.eda import create_report

st.set_option('deprecation.showPyplotGlobalUse', False)

class EDA:
    def __init__(self,data):
        # Initialize Dataset attributes
        self.data = data  # Placeholder for dataset
        self.columns=data.columns

    def format_report(self, report):

        formatted_report = ""
        for section, data in report.items():
            formatted_report += f"## {section}\n"
            if section == 'Basic Info':
                formatted_report += f"{data}\n\n"
            elif section == 'Descriptive Statistics' or section == 'Missing Values':
                if isinstance(data, pd.DataFrame):
                    formatted_report += f"{data.to_markdown(index=False)}\n\n"
                else:
                    formatted_report += f"{data}\n\n"
            else:
                formatted_report += f"{section}: {data}\n\n"
        return formatted_report

    def generate_eda_report(self):
    
        report = {}  # Initialize an empty dictionary to store EDA results

        # Basic info using info()
        info_buffer = StringIO()
        self.data.info(buf=info_buffer)
        info_str = info_buffer.getvalue()

        # info_str=pd.DataFrame(info_str)

        # Extracting relevant information from the info() output
        lines = info_str.split('\n')  # Splitting the output by newline characters
        basic_info = '\n'.join(lines[3:])  # Selecting relevant summary information, excluding the first 3 lines

        report['Basic Info'] = f"```{basic_info}```"  # Format basic_info as code block in Markdown
    
        # Descriptive Statistics
        describe_expander = st.expander("Descriptive Statistics")
        with describe_expander:
            describe = self.data.describe()
            report['Descriptive Statistics'] = describe.to_markdown(index="default")

        # Handling missing values
        missing_values_expander = st.expander("Missing Values")
        with missing_values_expander:
            missing_values = self.data.isnull().sum().to_frame(name='Missing_Values').reset_index()
            report['Missing Values'] = missing_values.to_markdown(index=False)

        return report

    def summary_statistics(self):
        report=self.generate_eda_report()
        summary=report.get('Descriptive Statistics')
        return summary

    def info_statistics(self):
        report=self.generate_eda_report()
        info=report.get('Basic Info')
        return info
    
    def missing_values(self):
        report=self.generate_eda_report()
        missing_values=report.get('Missing Values')
        return missing_values

    def bar_chart(self, column):
        st.write(f"### Bar Chart for {column}")
        chart_data = self.data[column].value_counts(dropna=False).reset_index()
        chart_data.columns = ['index', column + '_count']
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('index:N', title='Category', sort='-y'),  # Sort the bars by count
            y=alt.Y(column + '_count', title='Count')  # Specify the y-axis field and title
        )
        return chart

    def pie_chart(self, column):
        st.write(f"### Pie Chart for {column}")
        chart_data = self.data[column].value_counts(dropna=False).reset_index()
        chart_data.columns = ['index', column + '_count']
        total_count = chart_data[column + '_count'].sum()
        chart_data['percentage'] = (chart_data[column + '_count'] / total_count) * 100
        chart = alt.Chart(chart_data).mark_arc().encode(
            theta='percentage',
            color='index',
            tooltip=['index', 'percentage']
        ).properties(
            width=400,
            height=400
        )
        return chart

    def box_plot(self, column):
        try:
            self.data[column] = pd.to_numeric(self.data[column])
        except ValueError as e:
            st.write(f"Cannot convert {column} to a numeric type: {e}")
            return None

        if self.data[column].dtype not in ['int64', 'float64']:
            st.write(f"Column '{column}' is not numeric after conversion")
            return None

        # Calculate statistics
        median = self.data[column].median()
        quartiles = self.data[column].quantile([0.25, 0.75])
        mean = self.data[column].mean()
        std_dev = self.data[column].std()
        max_val = self.data[column].max()
        min_val = self.data[column].min()
        data_range = max_val - min_val
        iqr = quartiles[0.75] - quartiles[0.25]
        variance = self.data[column].var()

        # Create the box plot with annotations
        plt.figure(figsize=(8, 6))

        # Shifting the plot to the left side
        plt.subplots_adjust(right=0.7)  # Adjust the space on the right side of the plot
        
        self.data.boxplot(column=column)

        # Fill the boxplot with blue color
        bp = self.data.boxplot(column=column, patch_artist=True, boxprops=dict(facecolor='blue',alpha=0.3))

         # Get the current font size used for axis labels
        x_axis_fontsize = plt.gcf().axes[0].xaxis.get_ticklabels()[0].get_fontsize()
        y_axis_fontsize = plt.gcf().axes[0].yaxis.get_ticklabels()[0].get_fontsize()
        table_fontsize = min(x_axis_fontsize, y_axis_fontsize)  # Use the smaller font size

    
        # Define positions for the statistics table inside the plot
        table_data = [
            ['Mean', f'{mean:.2f}'],
            ['Standard Deviation', f'{std_dev:.2f}'],
            ['Variance', f'{variance:.2f}'],
            ['Maximum', f'{max_val:.2f}'],
            ['Minimum', f'{min_val:.2f}'],
            ['Range', f'{data_range:.2f}'],
            ['Median', f'{median:.2f}'],
            ['Q1', f'{quartiles[0.25]:.2f}'],
            ['Q3', f'{quartiles[0.75]:.2f}'],
            ['IQR', f'{iqr:.2f}']
        ]

        # # Draw a horizontal dashed line as a separator
        # plt.axhline(y=mean, color='black', linestyle='--',
        #              linewidth=1.5)  
        
        # Add a table inside the plot at the top-left corner
        
        table=plt.table(cellText=table_data,
                loc='upper left',
                cellLoc='center',
                colWidths=[1, 0.5],
                colLabels=['Statistic', 'Value'],
                bbox=[0.65, 0.4, 0.3, 0.5], # Adjust position and size
                fontsize=50) # Set the font size for the table

        st.pyplot()  # Show the plot with the table inside
        matplotlib.pyplot.close()

    def bar_plot(self, column):
        # Check if the column is numeric, and if not, try converting it to a numeric type
        if self.data[column].dtype not in ['int64', 'float64']:
            try:
                self.data[column] = pd.to_numeric(self.data[column])
            except ValueError as e:
                st.write(f"Cannot convert {column} to a numeric type: {e}")
                return None

        # Create a bar plot using matplotlib directly
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        self.data[column].value_counts().plot(kind='bar')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f"Bar Plot for {column}")

        # Show the plot in Streamlit
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        matplotlib.pyplot.close()

    def count_plot(self, column):
        # Create a count plot using Seaborn
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed

        sns.countplot(data=self.data, x=column)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f"Count Plot for {column}")

        # Show the plot in Streamlit
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        matplotlib.pyplot.close()

    def pie_plot(self, column):
        # Check if the column is categorical
        if self.data[column].dtype not in ['object', 'category']:
            st.write(f"Cannot create a pie plot for non-categorical data in {column}")
            return None

        # Create a pie plot using matplotlib directly
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        self.data[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f"Pie Plot for {column}")

        # Show the plot in Streamlit
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        matplotlib.pyplot.close()
        
    def generate_visual_eda_report(self):
        report = create_report(self.data)
        report.show()
        
    def value_table(self, column):
        st.write(f"### Value Table for {column}")
        st.write(self.data[[column]].head(10))

    def correlation(self):
        return self.data.corr()

    def get_correlation_heatmap(self):
        st.write("### Correlation Matrix")
        numeric_columns = self.data.select_dtypes(include=['number'])  # Include all numeric types
        corr_matrix = numeric_columns.corr()

        # Melting the correlation matrix to have columns as both x and y variables
        corr_melted = pd.melt(corr_matrix.reset_index(), id_vars='index', var_name='column')

        # Creating the heatmap using Altair
        heatmap = alt.Chart(corr_melted).mark_rect().encode(
            x=alt.X('index:O', title=''),
            y=alt.Y('column:O', title=''),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='viridis'))
        ).properties(
            width=900,
            height=900
        )

        return heatmap

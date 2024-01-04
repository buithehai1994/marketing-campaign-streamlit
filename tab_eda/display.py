import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from tab_eda.logics import EDA
import pandas as pd
from dataprep.eda import create_report
import streamlit_pandas_profiling 
from streamlit_pandas_profiling import st_profile_report

def display_summary_statistics(dataset):
    eda=EDA(dataset)
    statistics_report=eda.summary_statistics()
    st.write(statistics_report)

def display_info(dataset):
    eda=EDA(dataset)
    info_report=eda.info_statistics()
    st.write(info_report)

def display_missing_values(dataset):
    eda=EDA(dataset)
    info_missing_values=eda.missing_values()
    st.write(info_missing_values)

def display_plots(dataset):
    eda = EDA(dataset)
    columns_list = list(dataset.columns)
    
    for idx, column in enumerate(columns_list):
        st.write(f"### Plots for {column}")
         # Check if the column is numeric or non-numeric
        if dataset[column].dtype in ['int64', 'float64']:  # Numeric data
            plot_type = st.radio(f"Select plot type for {column}:",
                                 ("Box Plot", "Bar Plot"),
                                 key=f"{column}_radio_{idx}")  # Unique key for each radio button
            
            if plot_type == "Box Plot":
                eda.box_plot(column)
            elif plot_type == "Bar Plot":
                eda.bar_plot(column)
        else:  # Non-numeric data
            plot_type = st.radio(f"Select plot type for {column}:",
                                 ("Pie Plot", "Count Plot"),
                                 key=f"{column}_radio_{idx}")
            if plot_type == "Pie Plot":
                eda.pie_plot(column)
            elif plot_type == "Count Plot":
                eda.count_plot(column)

def display_generate_eda_report(data):
    eda = EDA(data)
    profile = eda.generate_visual_eda_report()
    
    if profile is not None:
        # Get the HTML content from the profiling report
        html_content = profile.to_html()

        st.markdown('<h1>EDA Report</h1>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.error("There was an issue generating the EDA report.")

def display_generate_visual_eda_report(data):
    eda = EDA(data)
    profile = eda.generate_visual_eda_report()
    
    # Get the HTML content from the profiling report
    html_content = profile.to_html()

    st.markdown('<h1>EDA Report</h1>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(html_content, unsafe_allow_html=True)
    
# Function to generate and display word cloud for a given text
def display_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

def display_plot_distribution(df, obj, ordered_obj,title):
    ordered_months = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    obj_percentage = (df.groupby(['month', obj]).size() / df.groupby('month').size()).unstack(fill_value=0)[ordered_obj].reindex(index=ordered_months)
    
    custom_colors = plt.cm.get_cmap('Set3', len(ordered_obj))(range(len(ordered_obj)))

    plt.figure(figsize=(8, 6))
    ax1 = obj_percentage.plot(kind='bar', stacked=True, color=custom_colors)
    ax1.set(xlabel='Month', ylabel='Percentage', title=f'{obj} Distribution (Percentage by Month)')
    ax1.legend(title=str(obj), bbox_to_anchor=(1.05, 1), loc='upper left')

    for month in df['month'].unique():
        ax1.axvline(x=ordered_months.index(month), color='gray', linestyle='--', alpha=0.5)

    # Limit the y-axis height
    ax1.set_ylim(0, 1)  # Set the desired y-axis limits

    plt.title(title, fontsize=10)
    st.pyplot()
            
def display_correlation_heatmap(dataset):
    eda = EDA(dataset)
    correlation_heatmap = eda.get_correlation_heatmap()
    st.altair_chart(correlation_heatmap, use_container_width=True)
    explanation_text = """
    
    The correlation heatmap above illustrates the relationship between variables in the dataset. Notably, it shows that the 'Target' variable has a low correlation with other features.

    For further analysis and modeling, the dataset will be structured with dependent variable 'y' and independent variables 'X' as follows:

    - **y = df_cleaned['Target']**
    - **X = df_cleaned.drop(['Target', 'ID'], axis=1)**

    This code segregates the 'Target' variable into 'y' and retains the remaining variables in 'X'. The 'ID' column is excluded. Subsequently, the dataset will be split into distinct sets with a 90-10 ratio:

    - **X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=0.1, random_state=42)**
    - **X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)**

    Due to differing scales between the target variable and other features, scaling becomes essential for effective model training. This process standardizes all 'X' features:

    - **from sklearn.preprocessing import StandardScaler**
    - **sc = StandardScaler()**
    - **X_train = sc.fit_transform(X_train)**
    - **X_test = sc.transform(X_test)**
    - **X_val = sc.transform(X_val)**

    """
    
    st.markdown(explanation_text)  # Display the explanatory text


def display_stack_bar_chart(df):
    # Plot the graph displaying relationship between interest rate and other variables
    ordered_months = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    data = {
            'CCI': df.groupby('month')['cons.conf.idx'].mean().values,
            'CPI': df.groupby('month')['cons.price.idx'].mean().values,
            'Euribor3M': df.groupby('month')['euribor3m'].mean().values,
            'EmpVarRate': df.groupby('month')['emp.var.rate'].mean().values,
            'SubVarRate': df.groupby('month')['y'].apply(lambda x: (x == 'yes').mean()),
            'default': df.groupby('month')['default'].apply(lambda x: (x == 'yes').mean())
    }
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax = [ax1]
    
    colors = ['tab:green', 'tab:orange', 'tab:red', 'tab:blue', 'tab:purple', 'black']
    y_labels = ['Consumer Confidence Index (CCI)', 'Consumer Price Index (CPI)', 
                'Euribor 3-Month Rate', 'Employees Variation Rate', 
                'Subscribe Variation Rate', 'Default Rate']
    
    markers = ['s', 'x', 'o', '^', 'd', 'v']
    
    for i, (label, color, marker) in enumerate(zip(data.keys(), colors, markers)):
        if i > 0:
            ax_new = ax1.twinx()
            ax_new.spines['right'].set_position(('outward', i * 60))
            ax.append(ax_new)
        ax[i].plot(ordered_months, data[label], marker=marker, color=color)
        ax[i].set_ylabel(y_labels[i], color=color)
        ax[i].tick_params(axis='y', labelcolor=color)
    
    plt.title('Default Rate, CPI, CCI, Euribor3M, and EmpVarRate by Month', fontsize=10)
    plt.xticks(range(len(ordered_months)), ordered_months, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    for month, month_position in zip(ordered_months, range(len(ordered_months))):
        ax1.axvline(x=month_position, color='gray', linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

    # Plot stacked bar chart for job
    ordered_job = ['admin.','technician', 'management','entrepreneur', 'services', 'blue-collar','student','self-employed', 'housemaid','retired', 'unemployed','unknown']
    display_plot_distribution(df,'job', ordered_job,title="Job Distribution (Percentage by month)")

    # Plot stacked bar chart for education
    ordered_education = ['university.degree','professional.course','high.school', 'basic.9y','basic.6y', 'basic.4y','illiterate','unknown']
    display_plot_distribution(df,'education', ordered_education,title="Education Distribution (Percentage by month)")

    # Plot for Credit default
    credit_default_df = df[df['default'] == 'yes']
    ordered_obj=['technician','unemployed']

    obj_grouped = credit_default_df.groupby(['month', 'job']).size()
    total_grouped = credit_default_df.groupby('month').size()
    obj_percentage = (obj_grouped / total_grouped).unstack(fill_value=0)[ordered_obj].reindex(index=ordered_months)
    
    custom_colors = plt.cm.get_cmap('Set3', len(ordered_obj))(range(len(ordered_obj)))
    
    plt.figure(figsize=(15, 4))
    ax1 = obj_percentage.plot(kind='bar', stacked=True, color=custom_colors)
    ax1.set(xlabel='Month', ylabel='Percentage', title='Job Distribution of people having credit default (Percentage by Month)')
    ax1.legend(title='Job', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for month in credit_default_df['month'].unique():
        ax1.axvline(x=ordered_months.index(month), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(plt)

def display_analysis(image_path):
    st.title("Economic Phases and Consumer Behavior Analysis")
    st.write(
        "Fluctuations in default rates, interest rates, education levels, and job choices are heavily influenced by the business cycle. "
        "A typical business cycle has four phases: expansion, peaking, contraction, and trough. Each phase will have different characteristics and impacts on consumers. "
        "In this section, we will explore the significant impact of economic factors on consumer behavior by interpreting the graphs presented in the preceding pages through the lens of business cycles."
    )
    # Display the image
    if Path(image_path).exists():
        st.image(image_path, caption="Business Cycle", use_column_width=True)
    else:
        st.error("Image not found.")

    st.markdown("<p style='text-align: center;'>(Sexton, 2010)</p>", unsafe_allow_html=True)
 
    st.markdown("### I. Five phases:")
    # Phase 1
    st.markdown("#### 1. March to April: Expansion Phase - Emerging Optimism")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: CPI, CCI, EURIBOR3M, and employee variation rate increased.\n"
        "This suggests an overall positive economic environment, potentially indicating increased consumer spending and business confidence.\n"
        "* Default Rate: The percentage number of people having default increased sharply.\n"
        "This might be due to various factors, such as changes in employment stability or financial difficulties for borrowers.\n"
        "* Occupation and Education Changes: The decrease in the percentage of people in admin, technician, retired, and university degree categories, "
        "along with the sharp increase in the percentage of blue-collar workers, could indicate shifts in the job market, potentially influenced by economic conditions or industry changes.\n"
        "* Subscribe: The average number of people subscribing to a Telecom plan decreased considerably.\n")
    
    st.write(
        "b. **Analysis:**\n"
        "During this period, the increase in economic indicators like the Consumer Price Index (CPI) and Consumer Confidence Index (CCI) is evidence of a surge in consumer optimism and spending, "
        "possibly pointing towards increased consumer spending and greater business confidence.\n"
        "However, the simultaneous increase in the default rate could be attributed to underlying financial difficulties faced by certain segments of the population despite the overall economic positivity. "
        "The occupation and education changes, including a decrease in admin, technician, retired, and university degree categories, "
        "along with a spike in blue-collar workers, may reflect a realignment of the job market due to economic influences.\n" 
        "The slight decrease in telecom subscriptions could be linked to financial caution among consumers, despite the overall positive economic environment.")

    # Phase 2
    st.markdown("#### 2. April to May: Peak Phase: Sentiment Shift")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: CCI and Subscribe variation rate increased sharply, while employee variation rate, CPI, EURIBOR3M collapsed.\n"
        "* Default Rate: The percentage number of people having default plunged to a low.\n"
        "* Occupation and Education Changes: The percentage of blue-collar workers peaked and the percentage of people holding university degrees bottomed out could reflect changes in employment patterns, possibly due to shifts in demand for certain skills or industries.\n"
        "* Subscribe: The average number of people subscribing to Telecom shot up.\n")
    
    st.write(
        "b. **Analysis:**\n"
        "The peak in the CCI along with the sharp decline in economic indicators like employee variation rate, CPI, and EURIBOR3M signifies a shift in sentiment.\n" 
        "This change could be attributed to external factors causing uncertainty. The contrasting trends in default rates (a plunge) and occupation/education percentages (a peak in blue-collar workers and a bottoming out of university degree holders) might suggest a complex interaction between economic shifts and individual financial decisions. It seems that the economy is going to boom again. The increase in telecom subscriptions during this phase could be a result of consumers’ confidence about the economy."
    )
    # Phase 3
    st.markdown("#### 3. May to June: Contraction Phase: Volatility Unleashed")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: CCI and Subscribe variation rate went into free fall, while employee variation rate, CPI and EURIBOR3M bounced back.\n"
        "* Default Rate: The percentage number of people having default remained at a low level.\n"
        "* Occupation and Education Changes: The percentage of blue-collar decreased slightly but it still maintained a high ratio. The percentage of admins increased moderately. The percentage of people holding university degrees went up slightly.\n"
        "* Subscribe: The average number of people subscribing to Telecom collapsed.\n")
    
    st.write(
        "b. **Analysis:**\n"
        "The plummeting CCI and rebounding indicators like employee variation rate and CPI highlight increased economic volatility. This period might be marked by uncertainty, leading to fluctuations in consumer spending and investment decisions. The increase in admin roles suggests a potential shift towards administrative or managerial positions, possibly driven by businesses restructuring in response to the economic environment. The rise in the percentage of people holding university degrees could reflect individuals seeking higher education to improve their job prospects in the uncertain economic climate. In such times, consumers could prioritize essential expenses over discretionary ones, possibly leading to a decline in new subscriptions or upgrades for telecom services."
    )
    # Phase 4
    st.markdown("#### 4. June to July: Trough Phase: Inflationary Respite")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: CPI peaked, and CCI increased slightly.\n"
        "* Default Rate: Number of people having default remained a low level.\n"
        "* Occupation and Education Changes: There was a slight drop in the percentage of blue-collar workers and a moderate increase in the percentage of admin people. The percentage of people holding a university degree was unchanged.\n"
        "* Subscribe: The average number of people subscribing increased moderately.\n")
    
    st.write(
        "b. **Analysis:**\n"
        "The peak in CPI suggests a brief inflationary period, possibly driven by increased demand or supply chain disruptions. The decline in defaults and the increase in the CCI might indicate improved financial conditions and consumer sentiment. The shift in job categories might reflect a temporary rebound in certain sectors, while the overall decrease in blue-collar workers could hint at the effects of changing economic conditions. The moderate increase in subscriptions despite the peak in CPI could be a result of individuals re-evaluating their priorities and realizing the value of the service after a period of reduced subscriptions. This might indicate that these services are viewed as essential or offering convenience and utility that consumers are unwilling to forego.\n"
    )

    st.write (
        "These are products or services that consumers consider vital and necessary for their daily lives, and they are often prioritized even during challenging economic times."
    )
    
    st.markdown("### 5. July to August: Trough Phase: Stability and Rebound")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: There is a significant decline in CPI, EURIBOR3M, and employee variation rate. This suggests potential economic contraction or stability.\n"
        "* Default Rate: The number of people experiencing defaults remains at a low level.\n"
        "* Occupation and Education Changes: There is a peak in admin, technician, and university degree holders, indicating certain sectors rebounding or stabilizing, potentially linked to the economic conditions.\n"
        "* Subscription: The average number of subscriptions increased rapidly.\n"
    )
    
    st.write(
        "b. **Analysis:**\n"
        "In August, individuals with credit defaults pursued professional courses and became technicians. Here are some possible reasons for this pattern:\n"
        "+ Income Instability: Technicians might have experienced income instability due to fluctuations in demand for their specific technical skills, potentially leading to challenges in meeting debt obligations.\n"
        "+ Mismatch of Education and Occupation: Those pursuing professional courses and becoming technicians might have faced a mismatch between their education and the job market's demands, possibly resulting in lower income and financial strain leading to credit defaults.\n"
        "+ Debt Accumulation: Individuals pursuing professional courses might have accumulated educational debt. Coupled with challenges in finding suitable employment, this debt burden could have caused financial stress and credit defaults.\n"
        "==> The decline in CPI and EURIBOR3M, along with increased confidence, suggests a period of relative stability or even slight contraction. The peak in admin, technician, and university degree holders might indicate specific sectors rebounding or stabilizing, potentially driven by an improved economic outlook.\n"
        "==> As economic indicators stabilize and show signs of recovery, consumer confidence could rebound, resulting in an increased willingness to invest in additional services like telecom subscriptions, as people feel more financially secure. Hence, the subscription numbers increased slightly.\n"
    )
    st.markdown("### 6. August to September: Recovery phase: Bouncing Back (A new cycle start to begin)")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: The rapid recovery of CPI, EURIBOR3M, employee variation rate. These indicators suggest improved consumer confidence, stable borrowing conditions, and potentially increased economic activity.\n"
        "* Default Rate: The number of people having default remains at a low level.\n"
        "* Occupation Changes: The drop in the percentage of people in the technician category could be a response to changes in demand for specific technical skills.\n"
        "* Subscribe: The average number of people subscribing dropped dramatically.\n"
    )
    st.write(
        "b. **Analysis:**\n"
        "The drop in technician roles might be due to a shift in demand for specific technical skills. The overall trends could signify a rapid adaptation to the changing economic landscape.\n"
        "During periods of economic uncertainty or downturns, people might cut back on non-essential spending. As the economy starts recovering, there's often a release of pent-up demand for goods and services that were deferred during tougher times. Subscriptions to services like telecom plans might fall into this category. People who postponed upgrading their plans or signing up for new services due to financial concerns might now feel more comfortable spending on these non-essential activities."
    )

    st.markdown("### 7. September to October: Recovery phase: “Cautious Optimism” ")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: The drop in CPI and employee variations might signify an improving economic situation, potentially indicating a decrease in financial stress and inflation. However, the default rate increases dramatically.\n"
        "* Default Rate: The number of people having default increase.\n"
        "* Occupation Changes: Stable job and education distribution might reflect a period of relative equilibrium in the job market and educational pursuits.\n"
        "* Subscribe: The average number of people subscribing increases slightly.\n"
    )
    st.write(
        "b. **Analysis:**\n"
        "This behavior is in line with the notion that during times of economic uncertainty or recovery, consumers often prioritize essential expenditures over discretionary ones. As individuals emerge from challenging economic periods, they may be cautious about additional spending, particularly on non-essential services like telecom subscriptions.\n"
        "+ Lagging Effects: Even though the recession might officially be over, its effects can linger. Some individuals, households, and businesses might still be dealing with the financial repercussions of the recession, which could result in delayed defaults or financial difficulties.\n"
        "+ Debt Accumulation: During an economic downturn, people and businesses might accumulate debt to manage through the tough times. These debts could become problematic after the immediate crisis has passed, leading to an increase in defaults. The intersection of lagging effects and debt accumulation illustrates that economic recovery is not a uniform experience for everyone. The broader economic trends can instill hope and optimism, but personal circumstances and financial burdens can introduce complexities that might slow down the process of regaining confidence.\n"
        "+ Pent-Up Demand: As individuals regain confidence in their finances, they might be more open to considering new telecom subscriptions or upgrading their existing plans. So the subscription numbers bounced back.\n"
        "The decreasing default rate and CPI could imply a less stressful financial environment and reduced inflationary pressures. The stable job and education distribution might suggest a period of equilibrium, where the initial shocks have subsided, and the economy is gradually finding its footing."   
    )
    st.markdown("### 8. October to November: Recovery phase: “Consolidating Stability” ")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: The continued decrease in the number of people having default, along with the collapsed EURIBOR3M and employee variation rate, suggests ongoing economic stability and improved financial conditions. These trends suggest that individuals are experiencing fewer financial difficulties and that credit markets are functioning more smoothly.\n"
        "* Default Rate: The number of people having default decrease.\n"
        "* Occupation Changes: The increase in the percentage of people in management roles could indicate growth or expansion in sectors requiring management positions. It's likely that businesses are regaining confidence and are investing in management capacities to drive growth and navigate the stabilized economic environment.\n"
        "* Subscribe: The average number of people subscribing increases dramatically\n")
    st.write(
        "b. **Analysis:**\n"
        "The continued decrease in the default rate and indicators like EURIBOR3M and employee variation rate indicates ongoing economic stability. The increase in management positions could signify growth in sectors requiring leadership roles, potentially indicating further expansion.\n"
        "As economic stability continues, consumers may be more receptive to long-term commitments like subscription plans. The company could emphasize the benefits of bundled services or loyalty programs to encourage customer retention and discourage churn. However, the subscriptions surprisingly dropped. The reason for it is the sudden increase of people having credit default.\n"
        "Regarding the total number of individuals experiencing defaults, these incidents primarily clustered in August and November. This clustering might be attributed to mature debt obligations. For more details, please refer to the 'Job distribution of people having credit default' graph.\n"
    )
    
    st.markdown("### 9. November to December: Recovery phase :“Year-End Resilience” ")
    st.write(
        "a. **Characteristics:**\n"
        "* Economic Indicators: Slight increases in CPI and Employee Variation Rate, along with the highest CCI level, could suggest strong economic sentiment and potential growth.\n"
        "* Default Rate: The stable number of people having default at a low level might indicate that the worst of the financial difficulties has passed.\n"
        "* Occupation Changes: It remains stable. There is a considerable decrease in the percentage of people being management and blue collar. This shift might imply changes in industry demand or a strategic workforce restructuring in response to evolving economic factors.\n"
         "* Subscribe: The average number of people subscribing increases slightly\n"
    )
    st.write(
        "b. **Analysis:**\n"
        "The slight increases in CPI and Employee Variation Rate, coupled with the highest CCI level, suggest a strong year-end sentiment and potential economic growth. The stable default rate might indicate that the worst financial challenges have passed. The changes in management positions could be reflective of ongoing shifts in demographics and professional roles.\n"
        "As economic conditions continue to improve, individuals may feel more confident in making non-essential expenditures, such as subscribing to telecom services. The slight increase in subscriptions suggests a growing willingness among consumers to invest in additional services"
    )
    st.markdown("### II. Conclusion:")
    st.markdown("### 1. Interest Rates and Economic Conditions:")
    st.write(
        "Interest rates, such as the EURIBOR3M, serve as a pivotal lever in economic policy, influencing borrowing costs and consumer behavior. The observed trends in your analysis demonstrate how changes in these rates can have cascading effects on various economic aspects. When interest rates rise, borrowing becomes more expensive, leading to reduced consumer spending and business investment. This could explain the collapse in economic indicators like the employee variation rate and the CPI during certain periods. Conversely, when interest rates are low, businesses are more likely to invest in expansion, potentially driving job creation and stimulating economic growth.\n"
    )
    st.markdown("### 2. Impact on Job Choices:")
    st.write(
        "Interest rates can greatly influence the demand for certain job categories. A surge in blue-collar workers suggests industries like manufacturing, construction, or logistics might be expanding during periods of economic positivity. This aligns with the notion that these sectors could be taking advantage of lower borrowing costs to fuel their growth. The decrease in admin and technician roles might be reflective of cost-cutting measures in response to tighter financial conditions, as well as automation or changes in business models.\n"
    )
    st.markdown("### 3. Impact on Education Choices:")
    st.write(
        "Interest rates, as a component of broader economic conditions, can shape individuals' decisions regarding education. During economic upswings, individuals might be more willing to invest in higher education to capitalize on growth prospects. This trend aligns with the decrease in university degree holders during periods of economic positivity, suggesting that people may prioritize immediate job opportunities over extended education. Conversely, during economic downturns or uncertainty, there might be a preference for shorter-term vocational training to quickly enter the workforce.\n"
    )
    st.markdown("### 4. Default Rates and Financial Stability:")
    st.write(
        "Fluctuations in default rates can provide insights into the financial health of borrowers. The increase in the number of people having default during certain periods might reflect the sensitivity of borrowers to changes in interest rates and economic stability. Higher interest rates could strain borrowers' ability to repay loans, contributing to a rise in default rates. This underscores the crucial role that interest rates play in determining the affordability of debt and its impact on individuals' financial well-being.\n"
    )
    st.markdown("### 5. Occupation and Education Changes:")
    st.write(
        "The patterns of occupation and education changes mirror the ebb and flow of economic conditions. When economic indicators peak, signaling confidence and growth, there's a corresponding increase in management and university degree holders, indicating opportunities for skilled roles. During economic volatility or decline, the emphasis on blue-collar workers might reflect industries prioritizing cost-effectiveness and immediate labor needs over higher-skilled roles. Similarly, reductions in technician roles could reflect adjustments made due to fluctuations in demand for technical expertise.\n"
    )
    st.markdown("### 6. Economic Sentiment and Decision-Making:")
    st.write(
        "The Consumer Confidence Index (CCI) provides insight into consumer sentiment, which is instrumental in shaping economic activity. Positive sentiment fosters spending, investment, and borrowing, all of which contribute to economic growth. When CCI peaks, as observed in some instances, it indicates that consumers are optimistic about the economic outlook and are more likely to engage in economic activities that drive growth.\n"
        )
    st.write("In conclusion, the interplay between interest rates, economic indicators, job choices, education decisions, default rates, and economic sentiment illustrates the complex web of relationships within an economy. These factors not only influence each other but also impact individuals' decisions and behavior. Understanding these dynamics is crucial for policymakers, businesses, and individuals alike, as they navigate the ever-changing landscape of economic conditions and their far-reaching effects.")

import streamlit as st
from tab_report.logics import EDA

def eda_report_tab(file_path=None):
    st.title("EDA Report Tab")

    # Load EDA report
    loaded_eda = EDA(data=file_path)

    if loaded_eda:
        # Display the loaded EDA report
        st.subheader("Loaded EDA Report:")
        eda=loaded_eda.perform_eda()
        st.write(eda)
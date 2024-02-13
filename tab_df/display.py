import streamlit as st

def display_tab_df_content(dataset):
    # Display original DataFrame
    st.subheader("Original DataFrame:")
    st.write(dataset.head_df())

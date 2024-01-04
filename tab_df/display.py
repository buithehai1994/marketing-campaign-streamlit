import streamlit as st

def display_tab_df_content(dataset):
    if dataset.data is None:
        st.error("No dataset provided or error in loading data")
        return

    # Display original DataFrame
    st.subheader("Original DataFrame:")
    st.write(dataset.head_df())

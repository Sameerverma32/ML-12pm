import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Market Basket Analysis")

st.title("Market Basket Analysis")
st.markdown("This app performs Market Basket Analysis using the Apriori algorithm.")

upload_file = st.file_uploader("Upload Groceries_dataset.csv", type=["csv"])
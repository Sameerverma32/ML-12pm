import streamlit as st 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules 

st.set_page_config(page_title="Market Basket Analysis") # with wide layout='wide'

st.title("Market Basket Analysis")

st.markdown("This app performs Market Basket Analysis using the Apriori algorithm")

upload_file = st.file_uploader("Upload Groceries_dataset.csv", type=["csv"])


if upload_file:
    #read the uploaded file
    df = pd.read_csv(upload_file)
    st.success("File uploaded successfully!")

    # show a preview of the data
    st.subheader("Data Preview")
    st.write(df.head())

    #check if required columns are present in the dataset
    if{'Member_number', 'Date', 'itemDescription'}.issubset(df.columns):
        df_grouped = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()
        
        # convert the grouped data into a list of transactions
        transactions = df_grouped['itemDescription'].tolist()

        #Apply Transaction Encoding to the transactions
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        st.success("Transactions prepared and encoded successfully!")

        #Add sidebar sliders to set parameter 
        st.sidebar.header("Parameters")
        min_support = st.sidebar.slider("Minimum Support", 0.001, 0.02, 0.002, step=0.001)
        min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.1, step=0.05)

        #Apply apriori algorithm to find frequent itemsets

        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if not rules.empty:
            rules_sorted = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False)
            st.subheader("Top 10 Association Rules")
            st.dataframe(rules_sorted.head(10).style.format({
                'support': '{:.3f}',
                'confidence': '{:.2f}',
                'lift': '{:.2f}'
            }))
        
            #Allow users to download all rules as csv 
            csv = rules_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Association Rules as CSV",
                data=csv,
                file_name='association_rules.csv',
                mime='text/csv'
            )
        else:
            st.warning("No association rules found.")
    else:
        st.error("Columns missing, Please Upload a file with the required columns")
else:
    st.info("file not uploaded or invalid file format")
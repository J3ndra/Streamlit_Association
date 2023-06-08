import streamlit as st
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.title('Aplikasi Web Data Mining Asosiasi!')

st.write("""
         # Dataset
         ### Grocery History
         """)

data = pd.read_csv('data.csv')

st.write(data)

st.write("""
         # Algorithm
         ### Apriori
         """)

st.write(f"### Data shape {data.shape}")

st.write("# Preprocessing")
products_list = [product.split(",") for product in data['products'].values]
te = TransactionEncoder()
te_data = te.fit_transform(products_list)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)
st.dataframe(df_encoded)

st.write("### Sum the data")
sums = df_encoded.sum().sort_values(ascending=False)
st.write(sums)

st.write("### Support Values")
support = df_encoded.mean().sort_values(ascending=False)
st.write(support)

st.write("### Eliminate Support Values Under 0,2")
filtered_support = support[support >= 0.2]
st.write(filtered_support)
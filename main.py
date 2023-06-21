import streamlit as st
import pandas as pd
import itertools
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

st.title('Aplikasi Web Data Mining Asosiasi!')

st.write("""
         # Dataset
         ##### Select dataset!
         """)

nama_algoritma = st.sidebar.selectbox(
    'Pilih Algoritma',
    ('Apriori', 'FP Growth'),
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write(data)

    st.write(f"##### Data shape {data.shape}")

    st.write("# Preprocessing")
    products_list = [product.split(",") for product in data['products'].values]
    te = TransactionEncoder()
    te_data = te.fit_transform(products_list)
    df_encoded = pd.DataFrame(te_data, columns=te.columns_)
    st.dataframe(df_encoded)

    st.write("##### Sum the data")
    sums = df_encoded.sum().sort_values(ascending=False)
    st.write(sums)

    # Custom Function
    def ar_iterations(data, num_iter=1, support_value=0.1, iterationIndex=None):
        # Next Iterations
        def ar_calculation(iterationIndex=iterationIndex):
            # Calculation of support value
            value = []
            for i in range(0, len(iterationIndex)):
                result = data.T.loc[iterationIndex[i]].sum()
                result = len(
                    result[result == data.T.loc[iterationIndex[i]].shape[0]]) / data.shape[0]
                value.append(result)
            # Bind results
            result = pd.DataFrame(value, columns=["Support"])
            result["index"] = [tuple(i) for i in iterationIndex]
            result['length'] = result['index'].apply(lambda x: len(x))
            result = result.set_index("index").sort_values(
                "Support", ascending=False)
            # Elimination by Support Value
            result = result[result.Support > support_value]
            return result

        # First Iteration
        first = pd.DataFrame(df_encoded.T.sum(
            axis=1) / df_encoded.shape[0], columns=["Support"]).sort_values("Support", ascending=False)
        first = first[first.Support > support_value]
        first["length"] = 1

        if num_iter == 1:
            res = first.copy()
        # Second Iteration
        elif num_iter == 2:
            second = list(itertools.combinations(first.index, 2))
            second = [list(i) for i in second]
            res = ar_calculation(second)
        # All Iterations > 2
        else:
            nth = list(itertools.combinations(
                set(list(itertools.chain(*iterationIndex))), num_iter))
            nth = [list(i) for i in nth]
            res = ar_calculation(nth)

        return res

    st.write("# Iteration")
    st.write("##### First Iteration")
    iteration1 = ar_iterations(df_encoded, num_iter=1, support_value=0.1)
    st.write(iteration1)

    st.write("##### Second Iteration")
    iteration2 = ar_iterations(df_encoded, num_iter=2, support_value=0.1)
    st.write(iteration2)

    st.write("##### Third Iteration")
    iteration3 = ar_iterations(df_encoded, num_iter=3, support_value=0.01,
                               iterationIndex=iteration2.index)
    st.write(iteration3)

    st.write("##### Fourth Iteration")
    iteration4 = ar_iterations(df_encoded, num_iter=4, support_value=0.01,
                               iterationIndex=iteration3.index)
    st.write(iteration4)

    st.write(f"""
             # Algorithm
             ##### {nama_algoritma}
             """)

    if nama_algoritma == 'Apriori':
        freq_items = apriori(df_encoded, min_support=0.1,
                             use_colnames=True, verbose=1)
    else:
        freq_items = fpgrowth(df_encoded, min_support=0.1,
                              use_colnames=True, verbose=1)

    st.write(freq_items.sort_values("support", ascending=False))

    st.write(f"##### {nama_algoritma} Top 5")
    st.dataframe(freq_items.head())

    st.write(f"##### {nama_algoritma} Bottom 5")
    st.dataframe(freq_items.tail())

    st.write("# Association Rules & Info")
    df_ar = association_rules(
        freq_items, metric="confidence", min_threshold=0.5)
    st.write(df_ar)

    st.write("##### Explanation")
    st.write("""
             1. antecedent support : memberitahu kita kemungkinan produk anteseden saja
             2. consequent support : memberitahu kita kemungkinan konsekuensi produk saja
             3. support            : memberitahu kita nilai dari kedua produk (Anteseden dan Konsekuen)
             4. confidence         : memberitahu kita indikasi seberapa sering aturan tersebut ditemukan benar.
             """)

    st.write("# Show table by filtering from Streamlit")
    support_value = st.slider("Select Support Value", 0.0, 1.0, 0.5, 0.05)
    confidence_value = st.slider(
        "Select Confidence Value", 0.0, 1.0, 0.5, 0.05)
    filtered_df = df_ar[(df_ar.support > support_value) &
                        (df_ar.confidence > confidence_value)]
    sorted_df = filtered_df.sort_values("confidence", ascending=False)
    st.write(sorted_df)

    # Create plot
    sorted_df['antecedents'] = sorted_df['antecedents'].apply(
        lambda x: ', '.join(list(x)))
    sorted_df['consequents'] = sorted_df['consequents'].apply(
        lambda x: ', '.join(list(x)))
    fig = px.bar(sorted_df, x='antecedents',
                 y='confidence', color='consequents')
    st.plotly_chart(fig)

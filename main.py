import streamlit as st
import pandas as pd
import itertools
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

st.write("### Find Support Values For Each Product")
support = df_encoded.mean().sort_values(ascending=False)
st.write(support)

st.write("### Eliminate Support Values Under 0.2")
filtered_support = support[support >= 0.2]
st.write(filtered_support)

st.write("### Find Support Values For Pair Product")
second_filtered_support = list(
    itertools.combinations(filtered_support.index, 2))
second_filtered_support = [list(i) for i in second_filtered_support]
# Sample of combinations
st.write(second_filtered_support[:25])
# Finding support values
value = []
for i in range(0, len(second_filtered_support)):
    temp = df_encoded.T.loc[second_filtered_support[i]].sum()
    temp = len(temp[temp == df_encoded.T.loc[second_filtered_support[i]
                                             ].shape[0]]) / df_encoded.shape[0]
    value.append(temp)
# Create a data frame
secondIteration = pd.DataFrame(value, columns=["Support"])
secondIteration["index"] = [tuple(i) for i in second_filtered_support]
secondIteration['length'] = secondIteration['index'].apply(lambda x: len(x))
secondIteration = secondIteration.set_index(
    "index").sort_values("Support", ascending=False)
# Elimination by Support Value
secondIteration = secondIteration[secondIteration.Support > 0.1]
st.write(secondIteration)


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


st.write("### First Iteration")
iteration1 = ar_iterations(df_encoded, num_iter=1, support_value=0.1)
st.write(iteration1)

st.write("### Second Iteration")
iteration2 = ar_iterations(df_encoded, num_iter=2, support_value=0.1)
st.write(iteration2)

st.write("### Third Iteration")
iteration3 = ar_iterations(df_encoded, num_iter=3, support_value=0.01,
                           iterationIndex=iteration2.index)
st.write(iteration3)

st.write("### Fourth Iteration")
iteration4 = ar_iterations(df_encoded, num_iter=4, support_value=0.01,
                           iterationIndex=iteration3.index)
st.write(iteration4)

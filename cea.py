import numpy as np
import pandas as pd
import streamlit as st

# Loading Data from a CSV File
data = pd.DataFrame(data=pd.read_csv('trainingdata.csv'))

# Separating concept features from Target
concepts = np.array(data.iloc[:, 0:-1])

# Isolating target into a separate DataFrame
target = np.array(data.iloc[:, -1])

def learn(concepts, target):
    '''
    learn() function implements the learning method of the Candidate elimination algorithm.
    Arguments:
        concepts - a data frame with all the features
        target - a data frame with corresponding output values
    '''

    # Initialise S0 with the first instance from concepts
    specific_h = concepts[0].copy()

    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    # The learning iterations
    for i, h in enumerate(concepts):

        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                # Change values in S & G only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Checking if the hypothesis has a negative target
        if target[i] == "No":
            for x in range(len(specific_h)):
                # For negative hypothesis change values only in G
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    # Find indices where we have empty rows, meaning those that are unchanged
    indices = [i for i, val in enumerate(general_h) if val == ['?'] * len(specific_h)]
    for i in indices:
        # Remove those rows from general_h
        general_h.remove(['?'] * len(specific_h))
    # Return final values
    return specific_h, general_h

# Streamlit app
st.title('Candidate-Elimination Algorithm')

st.subheader('Training Data')
st.write(data)

st.subheader('Candidate-Elimination Algorithm Execution')

# Execute the algorithm
s_final, g_final = learn(concepts, target)

st.subheader('Final Specific Hypothesis')
st.write(s_final)

st.subheader('Final General Hypotheses')

# Convert final general hypotheses to DataFrame for tabular display
g_final_df = pd.DataFrame(g_final, columns=data.columns[:-1])

st.write(g_final_df)

from readline import set_pre_input_hook
from sys import setprofile
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go






def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]
    
model = load_model('gbc_model')


st.image('1615502068951.jpeg', use_column_width=False, width=500)
st.title('The Red Wine Classificator App')
st.write('NOT FOR COMERCIAL USE, IS A TEST FOR WINE QUALITY.')



fixed_acidity = st.sidebar.slider(label = 'Fixed Acidity', min_value = 4.0, max_value = 16.0 , value = 10.0, step = 0.1)
volatile_acidity = st.sidebar.slider(label = 'Volatile Acidity', min_value = 0.0, max_value = 1.6 , value = 0.5, step = 0.01)
citric_acid = st.sidebar.slider(label = 'Citric Acid', min_value = 0.0, max_value = 1.0 , value = 0.5, step = 0.01)
residual_sugar = st.sidebar.slider(label = 'Residual Sugar', min_value = 0.0, max_value = 16.0 , value = 0.5, step = 0.01)
chlorides = st.sidebar.slider(label = 'Chlorides', min_value = 0.0, max_value = 0.61 , value = 0.5, step = 0.01)
sulphates = st.sidebar.slider(label = 'Sulphates', min_value = 0.0, max_value = 2.0 , value = 0.5, step = 0.01)
alcohol = st.sidebar.slider(label = 'Alcohol', min_value = 8.0, max_value = 15.0 , value = 0.5, step = 0.01)
free_sulfur_dioxide = st.sidebar.slider(label = 'Free Sulfur Dioxide', min_value = 0.0, max_value = 72.0 , value = 0.5, step = 0.1)
total_sulfur_dioxide = st.sidebar.slider(label = 'Total Sulfur Dioxide', min_value = 0.0, max_value = 290.0 , value = 0.5, step = 0.01)
density = st.sidebar.slider(label = 'Density', min_value = 0.0, max_value = 1.0 , value = 0.5, step = 0.01)
pH = st.sidebar.slider(label = 'pH', min_value = 0.0, max_value = 5.0 , value = 0.5, step = 0.01)




features = {'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity,'citric acid': citric_acid, 'residual sugar': residual_sugar,  
            'chlorides': chlorides,  'sulphates': sulphates, 'alcohol': alcohol, 'free sulfur dioxide': free_sulfur_dioxide, 'total sulfur dioxide': total_sulfur_dioxide, 'density': density, 'pH': pH}



df = pd.DataFrame(features, index = [0])
prediction = predict_quality(model, df)       
features_df  = pd.DataFrame([features])



st.table(features_df.T)

if st.button('Predict'):    
    prediction = predict_quality(model, features_df)    
    st.write(' Based on the score and chemical products, the quality of wine is '+ str(prediction))



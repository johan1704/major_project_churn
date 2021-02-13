# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:01:58 2020
@author: Johan Kouame Agouale
"""

import streamlit as st
import pandas as pd
import pickle
from PIL import Image

#pickle_in=open('xgboost.pkl', 'rb')
#classifier=pickle.load(pickle_in)

model = joblib.load('XGBoost_model.sav')
#image=Image.open('down.jpg')
#st.image(image,width=500)


st.markdown("<h1 style='text-align: center; color: red;'><strong>CUSTOMER CHURN PREDICTOR</strong></h1>", unsafe_allow_html=True) 

selectbox=st.sidebar.selectbox('Select operation to be performed',['Enter values for prediction'
                                                                   ,'View training dataset','View dataset analysis'])
if selectbox == 'Enter values for prediction':
    
    st.subheader("Enter the appropriate values for each field and press 'Predict' to get the result.")
    
    CreditScore=st.number_input('CreditScore', min_value= 1 , max_value = 850 , value =200)
    
    Geography=st.selectbox('Geography',['France','Spain','Germany'])
    if Geography=='France':
        Geography=1
    elif Geography=='Spain':
        Geography=2
    else:
        Geography=3
    
    Gender=st.selectbox('Gender',['Female','Male'])
    if Gender=='Female':
        Gender=0
    else:
        Gender=1
    
    Age=st.number_input('Age', min_value=1 , max_value=100, value=25)
    
    Tenure=st.number_input('Tenure(year in the company)', min_value = 1, max_value = 450, value= 1)
    
    Balance=st.text_input('Balance')
    
    NumberOfProducts=st.number_input('Number of product used', min_value=1 , max_value=5, value=1)
    
    HasCreditCard=st.selectbox('HasCreditCard',['yes','no'])
    if HasCreditCard=='yes':
        HasCreditCard=1
    else:
        HasCreditCard=0
    
    IsAnActiveMember=st.selectbox('IsAnActiveMember',['yes','no'])
    if IsAnActiveMember=='yes':
        IsAnActiveMember=1
    else:
        IsAnActiveMember=0
    
    Estimated_Salary=st.text_input('Estimated_Salary')    
    
    
    
    input_dict={'CreditScore':CreditScore, 'Geography': Geography, 'Gender':Gender, 'Age':Age, 'Tenure':Tenure, 'Balance':Balance, 'NumberOfProducts':NumberOfProducts, 
                'HasCreditCard':HasCreditCard, 'IsAnActiveMember':IsAnActiveMember, 'Estimated_Salary':Estimated_Salary}
    
    input_df=pd.DataFrame([input_dict])
    
    if st.button("Predict"):
        output=model.predict(input_df)
        prob=model.predict_proba(input_df)
        if int(output)==0:
            st.error('Not leaving! The probability of customer NOT leaving is {}%'.format(round(prob[0][0]*100,2)))
        else:
            st.error('The probability of customer leaving is {}%, please review his profile.'.format(round(prob[0][1]*100)))
    
if selectbox == 'View training dataset':
    data=pd.read_csv('Churn_Modelling.csv')
    st.dataframe(data)
    st.text("Description of the implicit columns:")
    st.text("HasCrCard:Has a credit card?")
    st.text("IsActiveMemeber: Is an active Member?")
    st.text("Tenure:How many years the customer has been in the company")
    st.text("Geography:The location of the customer(either France,Spain or Germany)")
    
            

if selectbox == 'View dataset analysis':
    graph=st.selectbox('Select graph to be displayed',['Churn target vs Gender','Churn target vs Age',
                                                       'Churn target vs Geography'])
                                                 
    if graph == 'Churn target vs Gender':
        image2=Image.open('Churn risk per gender.png')
        st.image(image2, width=800)
        st.text("It appears that female's customers exit more than male's customers")
    if graph == 'Churn target vs Age':
        image3=Image.open('churn decision vs age.png')
        st.image(image3,width=900)
        st.text("It appears that customers with age's range of 35 and 52 exit more than others customers")
    if graph =='Churn target vs Geography':
        image4=Image.open('churn risk per Geography.png')
        st.image(image4,width=800)
        st.text("It appears that customers located in germany tend to exit more ,followed by france's customers")

st.markdown("""
<style>
body {

    background-color: orange;
}

</style>
    """, unsafe_allow_html=True)
    
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#f5ee0a,#f5ee0a);
}
</style>
""",
    unsafe_allow_html=True,
)
        
st.markdown('<style> body {font-weight:bold;background-image: url("https://cdn.pixabay.com/photo/2017/10/29/09/51/background-2899263_960_720.jpg"); background-size:cover;}</style>',unsafe_allow_html=True)

import streamlit as st

st.title('IPL Win Predictor')   # Giving title of the website

teams=['Sunrisers Hyderabad', 'Mumbai Indians',
       'Royal Challengers Bangalore', 'Kolkata Knight Riders',
       'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals',
       'Delhi Capitals']

cities=['Kolkata', 'Delhi', 'Abu Dhabi', 'Chennai', 'Bangalore',
       'Chandigarh', 'Hyderabad', 'Johannesburg', 'Mumbai', 'Bengaluru',
       'Visakhapatnam', 'East London', 'Port Elizabeth', 'Jaipur',
       'Durban', 'Centurion', 'Dubai', 'Cape Town', 'Dharamsala', 'Pune',
       'Ahmedabad', 'Raipur', 'Ranchi', 'Cuttack', 'Guwahati',
       'Navi Mumbai', 'Indore', 'Sharjah', 'Kimberley', 'Nagpur',
       'Bloemfontein']

import pickle
pipe=pickle.load(open('pipe.pkl','rb'))   # "rb" means "read binary" mode

col1,col2 = st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select the bowling team',sorted(teams))

city=st.selectbox('Select the host city',sorted(cities))

target_runs=st.number_input('Target',min_value=0)

col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('Current Score',min_value=0)
with col4:
    over=st.number_input('Current Over',format="%0.1f",min_value=0.0,step=0.1,max_value=20.0)
with col5:
    wicket=st.number_input('Wicket',min_value=0,max_value=10)

runs_left=target_runs-score
balls_left=120-(over*6)
wickets_left=10-wicket
rrr=runs_left/(balls_left/6)

import pandas as pd
if st.button('Predict Probability'):
    #pass
    crr=score/over
    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets_left],'target_runs':[target_runs],'crr':[crr],'rrr':[rrr]})
    #st.table(input_df)
    
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team+" : "+str(round(win*100))+"%")
    st.header(bowling_team+" : "+str(round(loss*100))+"%")
import streamlit as st

st.title('การทดสอบเขียนเว็บด้วย Python')
st.header("Sumett Ampornsak")
st.subheader('สาขาเทคโนโลยีสารสนเทศ')
st.markdown("----")

col1, col2 = st.columns(2)
#col1.write("This is column 1")
#col2.write("This is column 2")
with col1:
    st.image('./pic/sumett_pic.jpg')
with col2:
    st.image('./pic/iris-flower-background.jpg')


html_1 = """
<div style="background-color:#85F733;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

import pandas as pd

dt=pd.read_csv('./data/iris.csv')
st.write(dt.head(10))

import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

st.title('การทดสอบเขียนเว็บด้วย Python')
st.header("Sumett Ampornsak")
st.subheader('สาขาเทคโนโลยีสารสนเทศ')
st.markdown("----")

col1, col2 = st.columns(2)
with col1:
    st.image('./pic/sumett_pic.jpg')
with col2:
    st.image('./pic/iris-flower-background.jpg')

html_1 = """
<div style="background-color:#85F733;padding:15px;border-radius:15px 15px 15px 15px;border-style:solid;border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv('./data/iris.csv')
st.write(dt.head(10))

dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

show_chart = st.checkbox("Show bar chart")
if show_chart:
    st.bar_chart(dx2)

html_2 = """
<div style="background-color:#FFBF00;padding:15px;border-radius:15px 15px 15px 15px;border-style:solid;border-color:black">
<center><h5>การทำนายคลาสดอกไม้</h5></center>
</div>
"""
st.markdown(html_2, unsafe_allow_html=True)
st.markdown("")

ptlen = st.slider("กรุณาเลือกข้อมูล petal.length", 0, 10)
ptwd = st.slider("กรุณาเลือกข้อมูล petal.width", 0, 10)

splen = st.number_input("กรุณาเลือกข้อมูล sepal.length")
spwd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):
    X = dt.drop('variety', axis=1)
    y = dt.variety
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = np.array([[ptlen, ptwd, splen, spwd]])
    st.write(Knn_model.predict(x_input))
    out = Knn_model.predict(x_input)

    if out[0] == "Setosa":
        st.image("./pic/Irissetosa1.jpg")
        st.header("Setosa")
    elif out[0] == "Versicolor":
        st.image("./pic/irisVersicolor.jpg")
        st.header("Versicolor")
    else:
        st.image("./pic/Irisvirginica.jpg")
        st.header("Verginiga")

    st.button("ไม่ทำนายผล")
else:
    st.button("ไม่ทำนายผล")

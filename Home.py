import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Set page title and configure layout
st.set_page_config(
    page_title="การทดสอบเขียนเว็บด้วย Python",
    page_icon=":tulip:",
    layout="wide",
)

# Page title and header
st.title('การทดสอบเขียนเว็บด้วย Python')
st.header("Sumett Ampornsak")
st.subheader('สาขาเทคโนโลยีสารสนเทศ')
st.markdown("----")

# Side-by-side image display
col1, col2 = st.columns(2)
with col1:
    st.image('./pic/sumett_pic.jpg', use_container_width=True)
with col2:
    st.image('./pic/iris-flower-background.jpg', use_container_width=True)

# Statistics box on the left side
st.markdown(
    """
    <div style="background-color:#85F733;padding:15px;border-radius:15px;border-style:solid;border-color:black">
        <center><h5 style="color:white;">สถิติข้อมูลดอกไม้</h5></center>
    </div>
    """, unsafe_allow_html=True
)

# Display first 10 rows of the dataset
dt = pd.read_csv('./data/iris.csv')
st.dataframe(dt.head(10))

# Summarize statistics
dt_stats = dt.describe()
st.write(dt_stats)

# Bar chart display with checkbox
show_chart = st.checkbox("Show bar chart")
if show_chart:
    st.bar_chart(dt_stats)

# Prediction box on the right side
st.markdown(
    """
    <div style="background-color:#FFBF00;padding:15px;border-radius:15px;border-style:solid;border-color:black">
        <center><h5 style="color:white;">การทำนายคลาสดอกไม้</h5></center>
    </div>
    """, unsafe_allow_html=True
)
st.markdown("")

# User input for prediction on the right side
ptlen = st.slider("กรุณาเลือกข้อมูล petal.length", 0, 10)
ptwd = st.slider("กรุณาเลือกข้อมูล petal.width", 0, 10)
splen = st.number_input("กรุณาเลือกข้อมูล sepal.length")
spwd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

# KNN prediction button on the right side
if st.button("ทำนายผล"):
    X = dt.drop('variety', axis=1)
    y = dt.variety
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = np.array([[ptlen, ptwd, splen, spwd]])
    predicted_class = Knn_model.predict(x_input)[0]

    # Display predicted class and associated image on the right side
    st.subheader(f"ผลทำนาย: {predicted_class}")
    if predicted_class == "Setosa":
        st.image("./pic/Irissetosa1.jpg", use_container_width=True)
    elif predicted_class == "Versicolor":
        st.image("./pic/irisVersicolor.jpg", use_container_width=True)
    else:
        st.image("./pic/Irisvirginica.jpg", use_container_width=True)

    st.button("ไม่ทำนายผล")
else:
    st.button("ไม่ทำนายผล")

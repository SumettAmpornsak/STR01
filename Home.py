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
st.sidebar.markdown(
    """
    <div style="background-color:#85F733;padding:15px;border-radius:15px;border-style:solid;border-color:black">
        <center><h5 style="color:white;">สถิติข้อมูลดอกไม้</h5></center>
    </div>
    """, unsafe_allow_html=True
)

# Display first 10 rows of the dataset in the sidebar
dt = pd.read_csv('./data/iris.csv')
st.sidebar.dataframe(dt.head(10))

# Summarize statistics in the sidebar
dt_stats = dt.describe()
st.sidebar.write(dt_stats)

# Bar chart display with checkbox in the main section
show_chart = st.sidebar.checkbox("Show bar chart")
if show_chart:
    st.sidebar.bar_chart(dt_stats)

# Prediction box on the right side
st.markdown(
    """
    <div style="background-color:#FFBF00;padding:15px;border-radius:15px;border-style:solid;border-color:black">
        <center><h5 style="color:white;">การทำนายคลาสดอกไม้</h5></center>
    </div>
    """, unsafe_allow_html=True
)
st.markdown("")

# User input for prediction in the main section
ptlen = st.slider("กรุณาเลือกข้อมูล petal.length", 0, 10)
ptwd = st.slider("กรุณาเลือกข้อมูล petal.width", 0, 10)
splen = st.number_input("กรุณาเลือกข้อมูล sepal.length")
spwd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

# KNN prediction button in the main section
if st.button("ทำนายผล"):
    X = dt.drop('variety', axis=1)
    y = dt.variety
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = np.array([[ptlen, ptwd, splen, spwd]])
    predicted_class = Knn_model.predict(x_input)[0]

    # Display predicted class and associated image in the main section
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

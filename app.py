import streamlit as st
import importlib
from modules import about_ml
from modules import about_nn
from modules import demo_ml
from modules import demo_nn

# 🔹 พาธรูปภาพ
image_path = "datasources/ai5.jpg"

# 🔹 โหลดโมดูลใหม่ (สำหรับพัฒนา)
importlib.reload(about_ml)
importlib.reload(about_nn)
importlib.reload(demo_ml)
importlib.reload(demo_nn)

# 🔹 CSS ปรับ Sidebar
st.markdown(
    """
    <style>
    .css-1d391kg { background-color: #f8f9fa !important; }
    .sidebar .sidebar-content { background-color: #ffffff; padding: 20px; }
    .menu-item {
        background-color: #ffffff;
        color: #333;
        padding: 12px;
        margin: 10px 0;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        cursor: pointer;
    }
    .menu-item:hover { background-color: #4CAF50; color: white; transform: scale(1.05); }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 จัดการ Session State
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "home"

# 🔹 Sidebar Menu
st.sidebar.markdown("<h2 style='text-align: center;'> Navigation</h2>", unsafe_allow_html=True)

pages = {
    " Home": "home",
    " About ML": "about_ml",
    " Demo ML": "demo_ml",
    " About NN": "about_nn",
    " Demo NN": "demo_nn"
}

for page_name, page_key in pages.items():
    if st.sidebar.button(page_name):
        st.session_state.selected_page = page_key

# 🔹 Render Pages
if st.session_state.selected_page == "home":
    st.markdown("<h1 style='color: #4CAF50; text-align: center;'>Welcome to AI Project!</h1>", unsafe_allow_html=True)
    st.subheader("Explore Machine Learning and Neural Networks")
    st.write("This project demonstrates how Machine Learning and Neural Networks work with interactive visualizations and real-world examples.")

    # 🔹 **แสดงรูปภาพ**
    st.image(image_path, caption="AI Concept Image", use_container_width=True)

    st.subheader("Get started by selecting a topic from the sidebar!")

elif st.session_state.selected_page == "about_ml":
    about_ml.show()

elif st.session_state.selected_page == "demo_ml":
    demo_ml.show()

elif st.session_state.selected_page == "about_nn":
    about_nn.show()

elif st.session_state.selected_page == "demo_nn":
    demo_nn.show()

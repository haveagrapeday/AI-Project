import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def show():
    st.title("📖 About Machine Learning in Harry Potter Dialogue Analysis")
    st.write("""
        This section explains the development process of the Machine Learning model used 
        to analyze dialogues from the Harry Potter movies.
    """)
    st.image("datasources/300.gif", use_container_width=600)
  
    st.subheader("🔹 Data Preparation")
    st.write("""
        - **Dataset Source**: Kaggle (Harry Potter Dialogue Dataset)
        - **Data Files Used**: 
            - `Dialogue.csv` - Contains dialogues spoken by characters.
            - `Characters.csv` - Contains metadata about characters.
        - **Data Cleaning**:
            - Removing missing/null values
            - Standardizing text format
            - Merging datasets using `Character_ID`
    """)

    # ML Algorithm Theory
    st.subheader("🔹 ML Algorithm Theory")
    st.write("""
        - **Algorithm Used**: Natural Language Processing (NLP) with **TF-IDF + Classification Model**.
        - **Why NLP?**:
            - Dialogues are text-based, requiring text processing techniques.
            - Term Frequency-Inverse Document Frequency (TF-IDF) helps extract important words.
        - **Classification Model**:
            - Decision Tree / Random Forest / SVM are considered for classifying dialogue patterns.
    """)

    # ML Model Development Steps
    st.subheader("🔹 ML Model Development Steps")
    st.write("""
        1. **Load and Clean Data**: Read CSV files, clean missing values, and merge datasets.
        2. **Text Preprocessing**: Tokenization, removing stopwords, and applying TF-IDF.
        3. **Train/Test Split**: Split data into training (80%) and testing (20%) sets.
        4. **Train the Model**: Use a classification algorithm like Random Forest.
        5. **Evaluate Performance**: Measure accuracy, precision, recall using confusion matrix.
    """)

    # Conclusion
    st.subheader("🔹 Conclusion")
    st.write("""
        - The dataset contains valuable insights into character dialogues.
        - NLP techniques help extract meaningful patterns from text.
        - A classification model can help determine which character a dialogue belongs to.
        - Further improvements can include deep learning approaches like transformers (BERT).
    """)
    
# 🔹 1. Load CSV files
    st.subheader("📌 Load Data Files")
    code_load = '''
import pandas as pd

df_dialogue = pd.read_csv("datasources/Harry_Potter_Movies/Dialogue.csv", encoding="latin1")
df_characters = pd.read_csv("datasources/Harry_Potter_Movies/Characters.csv", encoding="latin1")
df_students = pd.read_csv("datasources/Harry_Potter_Movies/harry_potter_1000_students.csv", encoding="latin1")
'''
    st.code(code_load, language="python")
    
    data_path = "datasources/Harry_Potter_Movies"
    files = ["Dialogue.csv", "Characters.csv"]
    dataframes = {}
    
    for file in files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            dataframes[file] = pd.read_csv(file_path, encoding="latin1")
        else:
            st.error(f"❌ {file} not found. Please check the file path.")
            return

    df_dialogue = dataframes["Dialogue.csv"]
    df_characters = dataframes["Characters.csv"]
    df_students = pd.read_csv("datasources/Harry_Potter_Movies/harry_potter_1000_students.csv", encoding="latin1")

    # 🔹 2. Clean column names
    st.subheader("📌 Clean Column Names")
    for df in [df_dialogue, df_characters, df_students]:
        df.columns = df.columns.str.replace(" ", "_").str.strip()

    # 🔹 3. Display columns of each file
    st.subheader("🔍 Columns in Each File")
    for name, df in dataframes.items():
        st.write(f"**{name}:**", list(df.columns))

    # 🔹 4. Merge Dialogue.csv + Characters.csv
    if "Character_ID" in df_dialogue.columns and "Character_ID" in df_characters.columns and "Character_Name" in df_characters.columns:
        st.subheader("📌 Merge Dialogue and Characters Data")
        df = df_dialogue.merge(df_characters, on="Character_ID", how="left")
        st.write("**🔍 Sample of Merged Data:**")
        st.write(df.head())


    st.image("datasources/200.gif", use_container_width=True)



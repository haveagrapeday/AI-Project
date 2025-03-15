import streamlit as st

def show():
    st.title("📖 About Machine Learning in Harry Potter Dialogue Analysis")
    st.write("""
        This section explains the development process of the Machine Learning model used 
        to analyze dialogues from the Harry Potter movies.
    """)
    st.image("datasources/300.gif", use_container_width=600)
    # Data Preparation
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
    
    st.image("datasources/200.gif", use_container_width=True)

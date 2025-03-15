import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show():
    st.title("📊 Machine Learning Demo")
    st.write("This page displays data and basic analysis for Machine Learning.")

    # 🔹 1. Load CSV files
    st.subheader("📌 Load Dialogue.csv and Characters.csv")
    code_load = '''
import pandas as pd

df_dialogue = pd.read_csv("datasources/Harry_Potter_Movies/Dialogue.csv", encoding="latin1")
df_characters = pd.read_csv("datasources/Harry_Potter_Movies/Characters.csv", encoding="latin1")
'''
    st.code(code_load, language="python")
    
    data_path = "datasources/Harry_Potter_Movies"
    dialogue_path = os.path.join(data_path, "Dialogue.csv")
    characters_path = os.path.join(data_path, "Characters.csv")
    
    if not os.path.exists(dialogue_path) or not os.path.exists(characters_path):
        st.error("❌ CSV files not found. Please check the file paths.")
        return
    
    df_dialogue = pd.read_csv(dialogue_path, encoding="latin1")
    df_characters = pd.read_csv(characters_path, encoding="latin1")

    # 🔹 2. Clean column names
    st.subheader("📌 Clean Column Names")
    code_clean = '''
df_dialogue.columns = df_dialogue.columns.str.replace(" ", "_").str.strip()
df_characters.columns = df_characters.columns.str.replace(" ", "_").str.strip()
'''
    st.code(code_clean, language="python")

    df_dialogue.columns = df_dialogue.columns.str.replace(" ", "_").str.strip()
    df_characters.columns = df_characters.columns.str.replace(" ", "_").str.strip()

    # 🔹 3. Display columns of each file
    st.subheader("🔍 Columns in Each File")
    st.write("**Dialogue.csv:**", list(df_dialogue.columns))
    st.write("**Characters.csv:**", list(df_characters.columns))

    # 🔹 4. Merge Dialogue.csv + Characters.csv
    if "Character_ID" in df_dialogue.columns and "Character_ID" in df_characters.columns and "Character_Name" in df_characters.columns:
        st.subheader("📌 Merge Data from Both Files")
        code_merge = '''
df = df_dialogue.merge(df_characters, on="Character_ID", how="left")
'''
        st.code(code_merge, language="python")
        
        df = df_dialogue.merge(df_characters, on="Character_ID", how="left")

        # Show merged data sample
        st.write("**🔍 Sample of Merged Data:**")
        st.write(df.head())

        # 🔹 5. Plot dialogue count
        st.subheader("📊 Character Dialogue Count")
        code_plot = '''
char_counts = df["Character_Name"].value_counts().head(10)

fig, ax = plt.subplots()
sns.barplot(x=char_counts.values, y=char_counts.index, palette="viridis", ax=ax)
ax.set_xlabel("Dialogue Count")
ax.set_ylabel("Character Name")
ax.set_title("Top 10 Characters with Most Dialogues")
st.pyplot(fig)
'''
        st.code(code_plot, language="python")

        char_counts = df["Character_Name"].value_counts().head(10)

        fig, ax = plt.subplots()
        sns.barplot(x=char_counts.values, y=char_counts.index, palette="viridis", ax=ax)
        ax.set_xlabel("Dialogue Count")
        ax.set_ylabel("Character Name")
        ax.set_title("Top 10 Characters with Most Dialogues")
        st.pyplot(fig)

        # 🔹 6. Select character to view dialogues
        st.subheader("🔍 Select a Character to View Dialogues")
        st.write("Since character names are in a separate file from dialogues, they are linked using 'Character_ID'.")

        code_select = '''
character_selected = st.selectbox("Select a Character", df["Character_Name"].dropna().unique())

st.subheader(f"📢 Dialogues of {character_selected}")
st.write(df[df["Character_Name"] == character_selected][["Dialogue"]].head(5))
'''
        st.code(code_select, language="python")

        character_selected = st.selectbox("Select a Character", df["Character_Name"].dropna().unique())

        st.subheader(f"📢 Dialogues of {character_selected}")
        st.write(df[df["Character_Name"] == character_selected][["Dialogue"]].head(5))
    else:
        st.error("❌ 'Character_ID' or 'Character_Name' column not found. Please check the CSV files.")

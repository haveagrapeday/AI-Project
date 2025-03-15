import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load data safely
def load_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding="latin1")
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {e}")
        return None

df_dialogue = load_csv("datasources/Harry_Potter_Movies/Dialogue.csv")
df_characters = load_csv("datasources/Harry_Potter_Movies/Characters.csv")
df_students = load_csv("datasources/Harry_Potter_Movies/harry_potter_1000_students.csv")

# 5. Plot dialogue count
if df_dialogue is not None and df_characters is not None:
    if "Character_ID" in df_dialogue.columns and "Character_ID" in df_characters.columns:
        df = df_dialogue.merge(df_characters, on="Character_ID", how="left")
        if "Character_Name" in df.columns:
            st.subheader("📊 Character Dialogue Count")
            char_counts = df["Character_Name"].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=char_counts.values, y=char_counts.index, palette="viridis", ax=ax)
            ax.set_xlabel("Dialogue Count")
            ax.set_ylabel("Character Name")
            ax.set_title("Top 10 Characters with Most Dialogues")
            st.pyplot(fig)
    else:
        st.error("❌ Column 'Character_ID' not found!")

# 7. Analyze Hogwarts House Traits
if df_students is not None:
    st.subheader("🏰 Hogwarts House Traits Analysis")
    traits = ["Bravery", "Intelligence", "Loyalty", "Ambition", "Dark_Arts_Knowledge", "Quidditch_Skills", "Dueling_Skills", "Creativity"]
    
    if all(trait in df_students.columns for trait in traits):
        house_means = df_students.groupby("House")[traits].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        house_means.T.plot(kind="bar", ax=ax)
        ax.set_title("Average Traits per Hogwarts House")
        ax.set_ylabel("Average Score")
        st.pyplot(fig)
    else:
        st.error("❌ Some required traits are missing from the dataset.")

# 8. Display Sample Data
if df_students is not None:
    st.subheader("🔍 Sample Data from Harry Potter Students")
    st.write(df_students.head())

# 9. Character Personality Traits Demo
st.subheader("🎭 Character Personality Traits")
characters = {
    "Harry Potter": ([8, 7, 6, 5, 2, 9, 6, 4], "Gryffindor"),
    "Hermione Granger": ([5, 10, 5, 4, 1, 2, 5, 8], "Gryffindor"),
    "Ron Weasley": ([6, 3, 8, 3, 2, 6, 4, 5], "Gryffindor"),
    "Draco Malfoy": ([3, 4, 2, 9, 7, 3, 5, 2], "Slytherin"),
    "Albus Dumbledore": ([9, 10, 8, 6, 5, 2, 10, 10], "Gryffindor"),
}

character_names = list(characters.keys())
selected_character = st.selectbox("Select a Main Character", character_names)
selected_traits, house = characters[selected_character]
traits = ["Bravery", "Intelligence", "Loyalty", "Ambition", "Dark Arts Knowledge", "Quidditch Skills", "Dueling Skills", "Creativity"]

fig, ax = plt.subplots()
sns.barplot(x=selected_traits, y=traits, palette="viridis", ax=ax)
ax.set_xlabel("Trait Score")
ax.set_ylabel("Trait")
ax.set_title(f"Personality Traits of {selected_character} ({house})")
st.pyplot(fig)

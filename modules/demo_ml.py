import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 โหลดข้อมูลอย่างปลอดภัย
def load_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding="latin1")
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {e}")
        return None

df_dialogue = load_csv("datasources/Harry_Potter_Movies/Dialogue.csv")
df_characters = load_csv("datasources/Harry_Potter_Movies/Characters.csv")
df_students = load_csv("datasources/Harry_Potter_Movies/harry_potter_1000_students.csv")

# 🔹 ตรวจสอบว่าไฟล์โหลดสำเร็จหรือไม่
if df_dialogue is not None and df_characters is not None:
    if "Character_ID" in df_dialogue.columns and "Character_ID" in df_characters.columns:
        df = df_dialogue.merge(df_characters, on="Character_ID", how="left")
    else:
        st.error("❌ Column 'Character_ID' not found in one of the datasets!")

# 🔹 5. Plot dialogue count
if df is not None and "Character_Name" in df.columns:
    st.subheader("📊 Character Dialogue Count")
    char_counts = df["Character_Name"].value_counts().head(10)

    fig, ax = plt.subplots()
    sns.barplot(x=char_counts.values, y=char_counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Dialogue Count")
    ax.set_ylabel("Character Name")
    ax.set_title("Top 10 Characters with Most Dialogues")
    st.pyplot(fig)
else:
    st.error("❌ Could not plot dialogue count due to missing data.")

# 🔹 7. Analyze Hogwarts House Traits
if df_students is not None:
    st.subheader("🏰 Hogwarts House Traits Analysis")
    traits = ["Bravery", "Intelligence", "Loyalty", "Ambition", "Dark_Arts_Knowledge", "Quidditch_Skills", "Dueling_Skills", "Creativity"]
    
    if all(trait in df_students.columns for trait in traits):
        house_means = df_students.groupby("House")[traits].mean()
        house_colors = {"Gryffindor": "#B22222", "Hufflepuff": "#FFD700", "Ravenclaw": "#4682B4", "Slytherin": "#2E8B57"}
        colors = [house_colors.get(house, "#808080") for house in house_means.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        house_means.T.plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Average Traits per Hogwarts House")
        ax.set_ylabel("Average Score")
        ax.legend(title="House")
        st.pyplot(fig)
    else:
        st.error("❌ Some required traits are missing from the dataset.")
else:
    st.error("❌ Could not analyze Hogwarts House traits due to missing data.")

# 🔹 8. Display Sample Data
if df_students is not None:
    st.subheader("🔍 Sample Data from Harry Potter Students")
    st.write(df_students.head())

# 🔹 9. Character Personality Traits Demo
st.subheader("🎭 Character Personality Traits")
characters = {
    "Harry Potter": ([8, 7, 6, 5, 2, 9, 6, 4], "Gryffindor"),
    "Hermione Granger": ([5, 10, 5, 4, 1, 2, 5, 8], "Gryffindor"),
    "Ron Weasley": ([6, 3, 8, 3, 2, 6, 4, 5], "Gryffindor"),
    "Draco Malfoy": ([3, 4, 2, 9, 7, 3, 5, 2], "Slytherin"),
    "Albus Dumbledore": ([9, 10, 8, 6, 5, 2, 10, 10], "Gryffindor"),
    "Rubeus Hagrid": ([5, 6, 6, 1, 4, 0, 5, 8], "Gryffindor"),
    "Severus Snape": ([6, 7, 10, 4, 8, 1, 7, 6], "Slytherin"),
    "Voldemort": ([3, 5, 1, 10, 10, 1, 8, 4], "Slytherin"),
    "Minerva McGonagall": ([8, 10, 9, 2, 2, 1, 5, 7], "Gryffindor"),
    "Luna Lovegood": ([6, 9, 8, 2, 1, 1, 2, 10], "Ravenclaw"),
    "Gilderoy Lockhart": ([2, 3, 1, 5, 3, 1, 3, 6], "Ravenclaw"),
    "Cedric Diggory": ([5, 2, 3, 7, 1, 8, 6, 3], "Hufflepuff"),
}

character_names = list(characters.keys())
selected_character = st.selectbox("Select a Main Character", character_names)
selected_traits, house = characters[selected_character]

trait_colors = ["#B22222", "#4682B4", "#FFD700", "#2E8B57", "#2E8B57", "#B22222", "#B22222", "#FFD700"]

traits = ["Bravery", "Intelligence", "Loyalty", "Ambition", "Dark Arts Knowledge", "Quidditch Skills", "Dueling Skills", "Creativity"]

fig, ax = plt.subplots()
sns.barplot(x=selected_traits, y=traits, palette=trait_colors, ax=ax)
ax.set_xlabel("Trait Score")
ax.set_ylabel("Trait")
ax.set_title(f"Personality Traits of {selected_character} ({house})")
st.pyplot(fig)

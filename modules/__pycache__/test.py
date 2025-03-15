import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ตัวอย่าง DataFrame สำหรับ df_dialogues และ df_students
df_dialogues = pd.DataFrame({
    "Character_Name": ["Harry Potter", "Hermione Granger", "Ron Weasley", "Draco Malfoy", "Harry Potter", "Hermione Granger"]
})

df_students = pd.DataFrame({
    "House": ["Gryffindor", "Gryffindor", "Gryffindor", "Slytherin", "Gryffindor", "Gryffindor"],
    "Bravery": [8, 5, 6, 3, 9, 5],
    "Intelligence": [7, 10, 3, 4, 10, 6],
    "Loyalty": [6, 5, 8, 2, 8, 6],
    "Ambition": [5, 4, 3, 9, 6, 1],
    "Dark_Arts_Knowledge": [2, 1, 2, 7, 5, 4],
    "Quidditch_Skills": [9, 2, 6, 3, 2, 0],
    "Dueling_Skills": [6, 5, 4, 5, 10, 5],
    "Creativity": [4, 8, 5, 2, 10, 8]
})

# 🔹 5. Plot dialogue count
st.subheader("📊 Character Dialogue Count")
char_counts = df_dialogues["Character_Name"].value_counts().head(10)

fig, ax = plt.subplots()
sns.barplot(x=char_counts.values, y=char_counts.index, palette="viridis", ax=ax)
ax.set_xlabel("Dialogue Count")
ax.set_ylabel("Character Name")
ax.set_title("Top 10 Characters with Most Dialogues")
st.pyplot(fig)

# 🔹 7. Analyze Hogwarts House Traits
st.subheader("🏰 Hogwarts House Traits Analysis")
traits = ["Bravery", "Intelligence", "Loyalty", "Ambition", "Dark_Arts_Knowledge", "Quidditch_Skills", "Dueling_Skills", "Creativity"]
house_means = df_students.groupby("House")[traits].mean()
house_colors = {"Gryffindor": "#B22222", "Hufflepuff": "#FFD700", "Ravenclaw": "#4682B4", "Slytherin": "#2E8B57"}
colors = [house_colors.get(house, "#808080") for house in house_means.index]

fig, ax = plt.subplots(figsize=(10, 6))
house_means.T.plot(kind="bar", ax=ax, color=colors)
ax.set_title("Average Traits per Hogwarts House")
ax.set_ylabel("Average Score")
ax.legend(title="House")
st.pyplot(fig)

# 🔹 8. Display Sample Data
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
    "Minerva McGonagall": ([8, 10, 9, 2, 2, 1, 5, 7], "Slytherin"),
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

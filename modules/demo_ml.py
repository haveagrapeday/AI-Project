import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def show():
    st.title("📊 Machine Learning Demo")
    st.write("This page displays data and basic analysis for Machine Learning.")

import pandas as pd

df_dialogue = pd.read_csv("datasources/Harry_Potter_Movies/Dialogue.csv", encoding="latin1")
df_characters = pd.read_csv("datasources/Harry_Potter_Movies/Characters.csv", encoding="latin1")
df_students = pd.read_csv("datasources/Harry_Potter_Movies/harry_potter_1000_students.csv", encoding="latin1")
    
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

  
    for df in [df_dialogue, df_characters, df_students]:
        df.columns = df.columns.str.replace(" ", "_").str.strip()

   
    st.subheader("🔍 Columns in Each File")
    for name, df in dataframes.items():
   


    st.subheader("📊 Character Dialogue Count")
    char_counts = df["Character_Name"].value_counts().head(10)



    # 🔹 6. Select character to view dialogues
    st.subheader("🔍 Select a Character to View Dialogues")
    character_selected = st.selectbox("Select a Character", df["Character_Name"].dropna().unique())
    st.subheader(f"📢 Dialogues of {character_selected}")
    st.write(df[df["Character_Name"] == character_selected][["Dialogue"]].head(5))

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
        "Minerva McGonagall": ([8, 10,9, 2, 2, 1, 5, 7], "Slytherin"),
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



    
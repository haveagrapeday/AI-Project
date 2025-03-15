import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show():
    st.title("📊 Machine Learning Demo")
    st.write("Find out which Hogwarts house you should be in! 🏰✨")

    # 👇 ส่วนของแบบทดสอบการเลือกบ้าน
    st.subheader("🏠 Hogwarts House Personality")
    questions = [
        ("What do you consider your most prominent quality?", ["Courage", "Intelligence", "Loyalty", "Ambition"]),
        ("What do you value most?", ["Justice", "Knowledge", "Friendship", "Power"]),
        ("If you saw a friend being bullied, what would you do?", ["Intervene immediately.", "Carefully plan a course of action before intervening.", "Try to resolve the situation peacefully through negotiation.", "Look for an opportunity to benefit from the situation."]),
        ("If you could learn one magical subject, which would you choose?", ["Defense Against the Dark Arts", "Potions", "Care of Magical Creatures", "Transfiguration"]),
        ("If you had to make a quick decision, what would you do?", ["Act immediately without hesitation.", "Analyze the situation thoroughly before deciding.", "Consider the impact on others before deciding.", "Decide based on your own best interests."]),
        ("What kind of place do you like the most?", ["Mountains, waterfalls", "Bookstores", "Cafes, workshops", "Hotels, bars"]),
        ("If you had the opportunity to be a leader, what would you do?", ["Make bold and decisive decisions.", "Use knowledge and wisdom.", "Be empathetic and listen to others' opinions.", "Lead the team towards success."]),
        ("If you had to work with someone you dislike, what would you do?", ["Complete the work without regard for the conflict.", "Think carefully and suggest a better way.", "Try to create a friendly and respectful atmosphere.", "Not cooperate and work alone, focusing on the results."])
    ]
    
    if "responses" not in st.session_state:
        st.session_state.responses = {}

    for idx, (q, options) in enumerate(questions):
        st.session_state.responses[idx] = st.selectbox(f"**{q}**", options, key=f"q{idx}")

    if st.button("🔮 Discover Your Hogwarts House!"):
        responses = list(st.session_state.responses.values())

        gryffindor = responses.count("Courage") + responses.count("Justice") + responses.count("Intervene immediately.") + responses.count("Defense Against the Dark Arts") + responses.count("Act immediately without hesitation.") + responses.count("Mountains, waterfalls") + responses.count("Make bold and decisive decisions.") + responses.count("Complete the work without regard for the conflict.")
        ravenclaw = responses.count("Intelligence") + responses.count("Knowledge") + responses.count("Carefully plan a course of action before intervening.") + responses.count("Potions") + responses.count("Analyze the situation thoroughly before deciding.") + responses.count("Bookstores") + responses.count("Use knowledge and wisdom.") + responses.count("Think carefully and suggest a better way.")
        hufflepuff = responses.count("Loyalty") + responses.count("Friendship") + responses.count("Try to resolve the situation peacefully through negotiation.") + responses.count("Care of Magical Creatures") + responses.count("Consider the impact on others before deciding.") + responses.count("Cafes, workshops") + responses.count("Be empathetic and listen to others' opinions.") + responses.count("Try to create a friendly and respectful atmosphere.")
        slytherin = responses.count("Ambition") + responses.count("Power") + responses.count("Look for an opportunity to benefit from the situation.") + responses.count("Transfiguration") + responses.count("Decide based on your own best interests.") + responses.count("Hotels, bars") + responses.count("Lead the team towards success.") + responses.count("Not cooperate and work alone, focusing on the results.")
        
        house_scores = {"Gryffindor": gryffindor, "Ravenclaw": ravenclaw, "Hufflepuff": hufflepuff, "Slytherin": slytherin}
        sorted_houses = sorted(house_scores.items(), key=lambda x: x[1], reverse=True)
        
        st.subheader(f"🏆 Your Hogwarts House is: {sorted_houses[0][0]}!")

    # 🔹 Load Data Files
    data_path = "datasources/Harry_Potter_Movies"
    students_file = os.path.join(data_path, "harry_potter_1000_students.csv")
    dialogues_file = os.path.join(data_path, "Dialogue.csv")
    
    if os.path.exists(students_file) and os.path.exists(dialogues_file):
        df_students = pd.read_csv(students_file, encoding="latin1")
        df_dialogues = pd.read_csv(dialogues_file, encoding="latin1")
        
        # 🔹 Clean column names
        df_students.columns = df_students.columns.str.replace(" ", "_").str.strip()
        df_dialogues.columns = df_dialogues.columns.str.replace(" ", "_").str.strip()
        
        # 🔹 5. Plot Dialogue Count
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
        
    else:
        st.warning("❌ Data files not found! Please ensure that the files 'harry_potter_1000_students.csv' and 'Dialogue.csv' are in the correct directory.")


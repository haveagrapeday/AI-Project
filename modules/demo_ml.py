import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show():
    st.title("📊 Machine Learning Demo")
    st.write("Find out which Hogwarts house you should be in! 🏰✨")


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
        
        house_scores = {"Gryffindor🦁": gryffindor, "Ravenclaw🦅": ravenclaw, "Hufflepuff🦡": hufflepuff, "Slytherin🐍": slytherin}
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
        
       

       
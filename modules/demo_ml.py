import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show():
    st.title("📊 Machine Learning Demo")
    st.write("ค้นหาว่าคุณควรอยู่บ้านไหนในฮอกวอตส์! 🏰✨")
    
    # 🔹 คำถามวิเคราะห์อุปนิสัย
    st.subheader("🏠 คำถามวิเคราะห์บ้านฮอกวอตส์")
    questions = [
        ("เมื่อเผชิญหน้ากับความท้าทาย คุณมักจะ?", ["เผชิญหน้าด้วยความกล้าหาญ", "วางแผนและใช้สติปัญญา", "ใช้ความภักดีและอดทน", "ใช้ความทะเยอทะยานเพื่อเอาชนะ"]),
        ("คุณให้ความสำคัญกับอะไรเป็นอันดับแรก?", ["ความกล้าหาญ", "สติปัญญา", "ความภักดี", "อำนาจและความสำเร็จ"]),
        ("สถานที่ในฮอกวอตส์ที่คุณอยากไปมากที่สุด?", ["สนามควิดดิช", "ห้องสมุด", "ห้องนั่งเล่นอันอบอุ่น", "ห้องแห่งความลับ"]),
        ("สัตว์วิเศษที่คุณอยากมีเป็นคู่หู?", ["สิงโต", "นกฮูก", "แบดเจอร์", "งู"])
    ]
    
    responses = []
    for q, options in questions:
        response = st.radio(q, options, key=q)
        responses.append(response)
    
    if st.button("🔮 ทำนายบ้านของคุณ!"):
        gryffindor = responses.count("เผชิญหน้าด้วยความกล้าหาญ") + responses.count("ความกล้าหาญ") + responses.count("สนามควิดดิช") + responses.count("สิงโต")
        ravenclaw = responses.count("วางแผนและใช้สติปัญญา") + responses.count("สติปัญญา") + responses.count("ห้องสมุด") + responses.count("นกฮูก")
        hufflepuff = responses.count("ใช้ความภักดีและอดทน") + responses.count("ความภักดี") + responses.count("ห้องนั่งเล่นอันอบอุ่น") + responses.count("แบดเจอร์")
        slytherin = responses.count("ใช้ความทะเยอทะยานเพื่อเอาชนะ") + responses.count("อำนาจและความสำเร็จ") + responses.count("ห้องแห่งความลับ") + responses.count("งู")
        
        house_scores = {"Gryffindor": gryffindor, "Ravenclaw": ravenclaw, "Hufflepuff": hufflepuff, "Slytherin": slytherin}
        sorted_houses = sorted(house_scores.items(), key=lambda x: x[1], reverse=True)
        
        st.subheader(f"🏆 บ้านของคุณคือ: {sorted_houses[0][0]}!")
        
        st.write("เลื่อนขวาเพื่อดูการวิเคราะห์บุคลิกภาพของบ้านแต่ละหลัง! ➡️")



    # 🔹 Load Data Files
    data_path = "datasources/Harry_Potter_Movies"
    df_students = pd.read_csv(os.path.join(data_path, "harry_potter_1000_students.csv"), encoding="latin1")

    # 🔹 Clean column names
    df_students.columns = df_students.columns.str.replace(" ", "_").str.strip()
    
    # 🔹 1. Analyze Hogwarts House Traits
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
    

  # 🔹 5. Plot dialogue count
    st.subheader("📊 Character Dialogue Count")
    char_counts = df["Character_Name"].value_counts().head(10)

    fig, ax = plt.subplots()
    sns.barplot(x=char_counts.values, y=char_counts.index, palette="viridis", ax=ax)
    ax.set_xlabel("Dialogue Count")
    ax.set_ylabel("Character Name")
    ax.set_title("Top 10 Characters with Most Dialogues")
    st.pyplot(fig)

    # 🔹 6. Select character to view dialogues
    st.subheader("🔍 Select a Character to View Dialogues")
    character_selected = st.selectbox("Select a Character", df["Character_Name"].dropna().unique())
    st.subheader(f"📢 Dialogues of {character_selected}")
    st.write(df[df["Character_Name"] == character_selected][["Dialogue"]].head(5))



    # 🔹 2. Display Sample Data
    st.subheader("🔍 Sample Data from Harry Potter Students")
    st.write(df_students.head())

    # 🔹 3. Character Personality Traits Demo
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

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
    
    if "responses" not in st.session_state:
        st.session_state.responses = {}

    for idx, (q, options) in enumerate(questions):
        st.session_state.responses[idx] = st.selectbox(f"**{q}**", options, key=f"q{idx}")

    if st.button("🔮 ทำนายบ้านของคุณ!"):
        responses = list(st.session_state.responses.values())

        gryffindor = responses.count("เผชิญหน้าด้วยความกล้าหาญ") + responses.count("ความกล้าหาญ") + responses.count("สนามควิดดิช") + responses.count("สิงโต")
        ravenclaw = responses.count("วางแผนและใช้สติปัญญา") + responses.count("สติปัญญา") + responses.count("ห้องสมุด") + responses.count("นกฮูก")
        hufflepuff = responses.count("ใช้ความภักดีและอดทน") + responses.count("ความภักดี") + responses.count("ห้องนั่งเล่นอันอบอุ่น") + responses.count("แบดเจอร์")
        slytherin = responses.count("ใช้ความทะเยอทะยานเพื่อเอาชนะ") + responses.count("อำนาจและความสำเร็จ") + responses.count("ห้องแห่งความลับ") + responses.count("งู")
        
        house_scores = {"Gryffindor": gryffindor, "Ravenclaw": ravenclaw, "Hufflepuff": hufflepuff, "Slytherin": slytherin}
        sorted_houses = sorted(house_scores.items(), key=lambda x: x[1], reverse=True)
        
        st.subheader(f"🏆 บ้านของคุณคือ: {sorted_houses[0][0]}!")
        st.write("เลื่อนขวาเพื่อดูการวิเคราะห์บุคลิกภาพของบ้านแต่ละหลัง! ➡️")

    # 🔹 โหลดข้อมูล
    data_path = "datasources/Harry_Potter_Movies"
    df_students = pd.read_csv(os.path.join(data_path, "harry_potter_1000_students.csv"), encoding="latin1")
    df_dialogues = pd.read_csv(os.path.join(data_path, "harry_potter_dialogues.csv"), encoding="latin1")
    
    df_students.columns = df_students.columns.str.replace(" ", "_").str.strip()
    df_dialogues.columns = df_dialogues.columns.str.replace(" ", "_").str.strip()
    
    # 🔹 วิเคราะห์ลักษณะบ้านฮอกวอตส์
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
    
    # 🔹 แสดงตัวอย่างข้อมูล
    st.subheader("🔍 Sample Data from Harry Potter Students")
    st.write(df_students.head())

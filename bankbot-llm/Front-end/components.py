import streamlit as st

def display_chat_history(chat_history):
    for sender, message in chat_history:
        if sender == "You":
            st.markdown(f"**🧍‍♂️ {sender}:** {message}")
        else:
            st.markdown(f"**🤖 {sender}:** {message}")

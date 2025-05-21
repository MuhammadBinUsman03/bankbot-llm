import streamlit as st
from services import get_llm_response, upload_document
from components import display_chat_history

# Page configuration
st.set_page_config(page_title="Banking LLM Assistant", layout="centered")

# Title and description
st.title("🏦 Banking Customer Support Assistant")
st.markdown("Ask your banking questions or upload related documents for help.")
st.divider()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input form
with st.form("chat_form"):
    st.subheader("💬 Chat with Assistant")
    user_input = st.text_input("Enter your question", placeholder="e.g., How to open a savings account?")
    submit_chat = st.form_submit_button("Ask")

    if submit_chat and user_input:
        with st.spinner("Getting answer from LLM..."):
            response = get_llm_response(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Assistant", response))

# Display conversation
if st.session_state.chat_history:
    st.markdown("### 🧠 Conversation")
    display_chat_history(st.session_state.chat_history)

    # Clear conversation button
    if st.button("🧹 Clear Conversation"):
        try:
            st.session_state.chat_history.clear()
            st.success("Conversation cleared successfully.")
        except Exception as e:
            st.error(f"Error clearing conversation: {e}")


st.divider()

# Document uploader
st.subheader("📄 Upload Document for Processing")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "json"])

if uploaded_file and st.button("📤 Upload"):
    with st.spinner("Uploading and processing..."):
        result = upload_document(uploaded_file)
        st.success("✅ Upload complete!")
        st.markdown(f"**Server Response:** {result}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit by Muhammad-AB | Powered by a fine-tuned LLaMA-3")
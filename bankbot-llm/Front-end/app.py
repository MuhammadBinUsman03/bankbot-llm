import streamlit as st
from services import get_llm_response, upload_document
from components import display_chat_history

# Page configuration
st.set_page_config(page_title="Banking LLM Assistant", layout="centered")

# Title and description
st.title("🏦 Banking Customer Support Assistant")
st.markdown("Ask your banking questions or upload JSON documents for help.")
st.divider()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = ""

# Collection name input
st.subheader("📚 Collection Name")
st.session_state.collection_name = st.text_input("Enter collection name", value=st.session_state.collection_name or "")

# Chat input form
with st.form("chat_form"):
    st.subheader("💬 Chat with Assistant")
    user_input = st.text_input("Enter your question", placeholder="e.g., How to open a savings account?")
    submit_chat = st.form_submit_button("Ask")

    if submit_chat:
        if not st.session_state.collection_name.strip():
            st.warning("⚠️ Please enter a collection name before chatting.")
        elif not user_input.strip():
            st.warning("⚠️ Please enter your question.")
        else:
            with st.spinner("Getting answer from LLM..."):
                response = get_llm_response(user_input, st.session_state.collection_name)
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
st.subheader("📄 Upload JSON Document for Processing")
uploaded_file = st.file_uploader("Choose a .json file", type=["json"])

if st.button("📤 Upload"):
    if not st.session_state.collection_name.strip():
        st.warning("⚠️ Please enter a collection name before uploading.")
    elif not uploaded_file:
        st.warning("⚠️ Please upload a JSON file.")
    elif uploaded_file.type != "application/json":
        st.error("❌ Only JSON files are supported.")
    else:
        with st.spinner("Uploading and processing..."):
            result = upload_document(uploaded_file, st.session_state.collection_name)
            st.success("✅ Upload complete!")
            st.markdown(f"**Server Response:** {result}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit by Muhammad-AB | Powered by a fine-tuned LLaMA-3")

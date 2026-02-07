import streamlit as st
import requests, os

API_URL = "http://127.0.0.1:8000/query"

# Set streamlit page config
st.set_page_config(page_title="AI-Powered Knowledge Assistant", page_icon="🤖")

# Title
st.title("AI-Powered knowledge assistant")

# Sidebar for file upload
st.sidebar.header("📜 Upload documents")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Save uploaded file
if uploaded_file:
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ {uploaded_file.name} uploaded succesfully!")

# Chat-like UI
st.subheader("Ask a question")
user_query = st.text_input("Type your question:")

# Send query to API
if st.button("Ask AI"):
    if user_query:
        response = requests.post(API_URL, json={"query": user_query})
        answer = response.json().get("response", "No response available.")
        st.markdown(f"**🤖 AI response:** {answer}")
    else:
        st.warning("Please enter a question.")
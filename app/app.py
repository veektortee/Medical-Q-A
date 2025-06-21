import sys
import os
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from retriever.medical_qa import generate_answer

st.set_page_config(page_title="Medical Q&A Assistant", layout="centered")

st.title("🩺 MediQuery")
st.write("Ask any health-related question below:")

# Track the last question and answer in session state
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""

question = st.text_area("Your Question", placeholder="e.g. What causes high blood pressure?")

if st.button("Get Answer") and question.strip():
    if question.strip() == st.session_state["last_question"]:
        st.info("You just asked this question.")
        st.success("Answer:")
        st.markdown(st.session_state["last_answer"])
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(question)
                st.session_state["last_question"] = question.strip()
                st.session_state["last_answer"] = answer
                st.success("Answer:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

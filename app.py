"""Streamlit entry point (starter UI only). Student 4 owns integration."""
import streamlit as st

st.set_page_config(page_title="ARTI303 Book Discovery", page_icon="📚", layout="centered")
st.title("📚 AI Book Discovery")
st.caption("ARTI303 Final Project — working title and prototype")
query = st.text_area("Describe the book you want to read", placeholder="A mysterious, atmospheric novel with complex characters...")
if st.button("Find books"):
    if not query.strip():
        st.warning("Please describe your reading preferences first.")
    else:
        st.info("The recommendation engine is not connected yet. The team will add real dataset-based results here.")
st.divider()
st.caption("Prototype UI — no AI recommendations or real book records are returned yet.")

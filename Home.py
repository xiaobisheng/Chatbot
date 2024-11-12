# This is The homepage script.
import streamlit as st

st.set_page_config(
    page_title="Chatbot Home Page",
    page_icon="🤖",
)

st.markdown(
    """
# CHATBOT HOME!
    
Welcome to my Gen AI Page!
    
There are the following chatbots I made using OPEN AI and langchain. 
    
- [✅] [General Chatbot]( /GeneralGPT): Casual chat.
- [✅] [Medical Chatbot]( /MedicalGPT): Enquire about medical information (please go to doctors for professional suggestions).
- [✅] [User Specific Chatbot]( /UserGPT)：upload your own files and chat.

Enjoy talking with them!
    
    """
)
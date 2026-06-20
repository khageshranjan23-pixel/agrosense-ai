import streamlit as st
st.write("Hello from test secrets app!")
try:
    st.write("Checking secrets...")
    exists = "key" in st.secrets
    st.write(f"Does key exist? {exists}")
except Exception as e:
    st.write(f"Caught exception of type {type(e)}: {e}")

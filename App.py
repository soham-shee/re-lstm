import streamlit as st
import time
import numpy as np
import pandas as pd

st.title("Load Forecasting Using LSTM")

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    if st.button("Train Model (Initial Training)"):
        st.switch_page("pages/Train.py")
with col2:
    if st.button("Re-train Model"):
        st.switch_page("pages/Retrain.py")
with col3:
    if st.button("Testing Model"):
        st.switch_page("pages/Test.py")
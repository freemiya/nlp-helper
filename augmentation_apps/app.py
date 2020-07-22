import streamlit as st
import json
import main

st.title("Data Augmentation")
@st.cache
def get_predictions():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        top_k = int(request.json['top_k'])
        res = main.get_mask_predictions(input_text, top_k)
        return json.dumps(res)
    except Exception as error:
        err = str(error)
        return json.dumps(err)

 
import streamlit as st
import json
import main

st.title("Data Augmentation")

# @st.cache
def get_predictions_colored(input_text: str, modelname:str, top_k= 5):
    """
    Input Text is given to multiple Language Models and predictions (with HTML tags)
    are collected.
    Args:
        input_text -> Text that will be masked and processed for prediction.
    Returns:
        res -> dictionary {
                            'bert': results,
                            'roberta': results,
                          }
    """
    try:
        res = main.get_mask_predictions(input_text, modelname, int(top_k))
        return res
    except Exception as error:
        err = str(error)
        return err

# https://discuss.streamlit.io/t/how-to-take-text-input-from-a-user/187/2
user_input = st.text_area("Input sentence", 'Company understands and accepts the failure of project.')
st.write("_Go ahead! <mask> your sentence. Otherwise, <mask> will be __randomly generated.__ _")

modelselected = st.sidebar.selectbox('Select a model',('roberta','electra','bert','xlmroberta','bart'))
topk = st.sidebar.slider('Top-k predictions', 0, 15, 5)

# https://discuss.streamlit.io/t/how-to-add-a-function-to-a-button/1478/3
if st.button('Generate!'):
    st.markdown(get_predictions_colored(user_input, modelselected, topk)[f'{modelselected}'], unsafe_allow_html = True)






import random
from random import choice
from string import ascii_uppercase, ascii_lowercase, ascii_letters, digits
import streamlit as st

def duplicate(yourchoice_string: str, num_times: int) -> str:
    """
    1. Consider the string as a list of characters. For ex: "abcdfm"
    2. Iterate for num_times, and in each iteration, pick a character randomly
    3. Stitch those characters for a random word generation.

    Usage:
        AU[0-9]+ is your regex.

    Args:
        yourchoice_string (str): String of all characters you want to consider
                                 Example: "abcdfm". Only get character from these characters
        num_times (int): Num of characters in the string you want to generate

    Returns:
        str: Characters stitched together into a word
    """
    return ''.join(choice(yourchoice_string) for i in range(num_times))

# https://discuss.streamlit.io/t/how-to-take-text-input-from-a-user/187/2
user_input = st.text_area("Input sentence with regex", '[0-9]_2||[A-Z]_4||532_3')
topk = st.sidebar.slider('Top-k predictions', 0, 15, 5)

def generate_combination(pattern: str):
    """
    Usage for user:
    Specify each sub-pattern separated by "||"
    For each sub-pattern, specify the [0-9]/[a-z]/[A-Z]/[aA-zZ] and
                          mention the num of chars to generate. Separate
                          these two by "_"
    Args:
        pattern (str): 
                       Example: [0-9]_2||[A-Z]_4||532_3
    Returns:
        str: Characters stitched together into a word
    """
    texts = []
    for ent in pattern.split("||"):
        et_type, et_num = ent.split("_")
        et_num = int(et_num)
        if et_type == "[0-9]":
            texts.append(duplicate(digits, et_num))
            print(duplicate(digits, et_num))
        elif et_type == "[A-Z]":
            texts.append(duplicate(ascii_uppercase, et_num))
            print(duplicate(ascii_uppercase, et_num))
        elif et_type == "[a-z]":
            texts.append(duplicate(ascii_lowercase, et_num))
            print(duplicate(ascii_uppercase, et_num))
        elif et_type == "[aA-zZ]":
            texts.append(duplicate(ascii_letters, et_num))
            print(duplicate(ascii_uppercase, et_num))
        elif et_type == "[Aa-Zz]":
            texts.append(duplicate(ascii_letters, et_num))
            print(duplicate(ascii_uppercase, et_num))
        else:
            texts.append(duplicate(et_type,et_num))
            print(duplicate(et_type,et_num))

    return "".join(texts)


# https://discuss.streamlit.io/t/how-to-add-a-function-to-a-button/1478/3
if st.button('Generate!'):
    for i in range(topk):
        st.write(generate_combination(user_input))
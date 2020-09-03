import re
from faker import Faker
import streamlit as st

fake = Faker("en_US")

testing_dict = {"country":fake.country,
                'person_male':fake.name_male,
               'person_female':fake.name_female,
                'address': fake.address,
                'city': fake.city,
                'company': fake.company,
                "country_code": fake.country_code,
                "language_name": fake.language_name,
               }

# https://discuss.streamlit.io/t/how-to-take-text-input-from-a-user/187/2
user_input = st.text_area("Input sentence with a template {requirement}", '{person_female} lives in {address}')
topk = st.sidebar.slider('#.of sentences to generate', 0, 15, 5)

def augment(sent, k=10):
    new_sents = []
    for _ in range(k):
        new_sents.append(sent.split())

    for idx, wrd in enumerate(sent.split()):
        matched = re.findall(r"{(.+)}",f"{wrd}")
        if  len(matched) > 0:
            matched = matched[0]
            for sent_id in range(k):
                new_sents[sent_id][idx] = testing_dict[matched]()

    new_sents = [' '.join(snt) for snt in new_sents]

    return new_sents

st.write("Look at sample patterns")
st.write(["{person_female} lives in {address}",
"{person_male} lives in the {city}",
"{person_female} lives in the {country}",
"{company} is natively from {city} , {country}",
"This survey is sponsored by {company} , {country_code}",
"Five {language_name} individuals are picked for this survey"])

st.write()

# https://discuss.streamlit.io/t/how-to-add-a-function-to-a-button/1478/3
if st.button('Generate!'):
    st.write(augment(user_input,topk))

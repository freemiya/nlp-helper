# augmentation_apps

## Vanilla Predictions
A basic application to check the language models' prediction. Supports `roberta`, `bert`, `bart`, `electra`, `xlmroberta`.

### Usage 

`pip install streamlit`

`streamlit run app.py`

### Application Details

Application will `<mask>` the sentence if not `<masked>`
![Prediction-1](imgs/testing_data2.PNG)
Updated UI.
![Prediction-2 updated](imgs/roberta.PNG)

## regex to words

`Usage for user:` 
* Specify each sub-pattern separated by "||".
* For each sub-pattern, specify the [0-9]/[a-z]/[A-Z]/[aA-zZ] and mention the num of chars to generate but separate these two by "_".


Example: `[0-9]_2||[A-Z]_4||532_3`
means 
1. Generate __2__ digits
2. Generate __4__ capital letters
3. Generate __3__ letters only using 5,3,2

![Regex 2 Word Image](imgs/regex2word.png)

## Grammar Module
_Work in Progress..._
## Class Transfer
_Work in Progress..._
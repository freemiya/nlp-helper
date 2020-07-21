import os
import pandas as pd
import streamlit as st
from typing import List, Union

import matplotlib.pyplot as plt

st.title('T_Automata_')

data_file=st.file_uploader("Choose a CSV file", type="csv")

if data_file is None:
    # Select a default file
    data_file = '../data/dell_total.csv'

rawdf = pd.read_csv(data_file)
dfcols = rawdf.columns

text_selected = st.selectbox('Select the text-column', dfcols)
tasklabel_selected = st.multiselect('Select the task label-columns', dfcols)

@st.cache # 👈 This function will be cached
def load_data(data_frame, textcol: str, labelcols: List[str]):
    """
    Read the csv file and return dataframe
    Args:
        data_frame -> Input file's dataframe
        textcol -> Name of the column you want to use for reviews/comments
        labelcols -> Name of the column(s) that act as labels
    Returns:
        A Pandas DataFrame
    """
    df = data_frame
    colnames = [f'{textcol}'] + [f'{colname}' for colname in labelcols]
    df = df[colnames]
    return df

def create_class_distribution(df: object, labelcols: List[str], class_separator: Union[str, None]):
    """
    Given a dataframe and names of label columns,
    get frequency of each label.
    Args:
        df -> Dataframe
        labelcols -> Name of the column(s) that act as labels
        class_separator -> There are datasets where one review/comment will have multiple labels
                            and they are separated by ";" or "," in a single column etc.
                            For example, label can be 
                            "confidence in brand;ease of use;quality;tech support"
    Returns:
        tasks_labels_distributions -> A Dictionary that stores another dictionary of labels
                                     i.e {
                                         'task1':
                                             {
                                                {
                                                    'label1': 50,
                                                    'label2': 70,
                                                }
                                             },
                                            'task2':
                                             {
                                                {
                                                    'label1': 80,
                                                    'label2': 90,
                                                }
                                             }
                                     }  
    """
    if class_separator is None: class_separator=';'

    # Create empty slots for the distribution plots to be filled
    plotspace = dict()
    for eachtask in labelcols:
        plotspace[eachtask] = st.empty()
    
    # For each task, store the dict that counts the distribution of labels
    tasks_labels_distributions = dict() 
    for eachtask in labelcols:
        task_label_s = df[f'{eachtask}'].values.tolist()
        # task_label_s ==> [[label1;label2], [label2]]
        task_label_distribution = dict()
        for task_labels in task_label_s:
            task_labels = str(task_labels)
            for task_label in task_labels.split(f'{class_separator}'):
                task_label_distribution[task_label] = task_label_distribution.get(task_label, 0) + 1

        # Get the image of the plot
        c = plot_pie(labels=task_label_distribution.keys(), \
                counts=task_label_distribution.values(),
                plot_name=eachtask)

        plotspace[eachtask] = st.pyplot(c)

        tasks_labels_distributions[eachtask] = task_label_distribution

    return tasks_labels_distributions

def plot_pie(labels: List[str], counts: List[int], plot_name: str):
    """
    Plot a pie.
    Args:
        labels -> List of class names
        counts -> List of class counts
        plot_name -> Usually, it is the Task Name
    Returns:
        the plot
    """
    plt.pie(counts,labels=labels,autopct='%1.1f%%')
    plt.title(f'{plot_name}')
    plt.axis('equal')

df = load_data(rawdf, text_selected, tasklabel_selected)

# If user wants to look at the data, this will let them see it
data_preview = st.checkbox('Look at the raw data')
if data_preview:
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    st.subheader('Raw look at the data')
    st.dataframe(df, width=1000)
    data_load_state.text('')

# If user wants to look at the data, this will let them see it
distribution_preview = st.checkbox('Look at class distribution')
if distribution_preview:
    tasks_labels_distributions = create_class_distribution(df, labelcols=tasklabel_selected, class_separator=';')
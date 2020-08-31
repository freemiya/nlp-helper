import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Union
from collections import Counter
from nltk.corpus import stopwords
import seaborn as sn
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

def create_class_distribution(df: object, taskcols: List[str], class_separator: Union[str, None]):
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
    # When there's no class-separator, it doesn't matter what separator you take
    if class_separator is None: class_separator=';'

    # Create empty slots for the distribution plots to be filled
    plotspace = dict()
    for eachtask in taskcols:
        plotspace[eachtask] = st.empty()
        
    # For each task, store the dict that counts the distribution of labels
    tasks_labels_distributions = dict()
    for eachtask in taskcols:
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
    Reference: https://datatofish.com/pie-chart-matplotlib/
    Plot a pie.
    Args:
        labels -> List of class names
        counts -> List of class counts
        plot_name -> Usually, it is the Task Name
    Returns:
        the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12,12))
    plt.pie(counts,labels=labels,autopct='%1.1f%%')
    plt.title(f'{plot_name}')
    plt.axis('equal')

def get_keywords_per_class(df: object, textcol: str, taskcols: List[str], class_separator=';', topkwords=200):
    """
    For venn diagrams: https://towardsdatascience.com/how-to-create-and-customize-venn-diagrams-in-python-263555527305
    For every row i.e sentence,
        Split the sentence into words (ignore the stopwords) and store them in a list
        For each task col,
            For each labelname in task_labels,
                word ->(mapped to) task_label
                task_label -> word

    Args:
        df -> Dataframe
        labelcols -> Name of the column(s) that act as labels
        class_separator -> There are datasets where one review/comment will have multiple labels
                            and they are separated by ";" or "," in a single column etc.
                            For example, label can be 
                            "confidence in brand;ease of use;quality;tech support"

    Returns:
        {
            'task1':
                {
                    'label1': {"word1":185,...}, # This is label_word_dist
                    'label2': {"word3":1,...}

                },
            'task2':
                {
                    'label4': {"word1":185,...}
                }
        }
    """ 
    num_rows = df.shape[0]
    stop_words = set(stopwords.words('english'))

    task_label_word_distribution = dict()

    # Write a progress bar
    processing_bar = st.progress(0)

    for idx in range(num_rows):
        
        sentence = str(df[f'{textcol}'][idx])
        if sentence=='nan':continue
        keywords = [word for word in sentence.split() if word not in stop_words]
        

        for eachtask in taskcols:
            task_label_s = str(df[f'{eachtask}'][idx])
            if task_label_s =='nan': continue
            # task_label_s ==> label1;label2
            label_word_distribution = task_label_word_distribution.get(eachtask, dict())

            for task_label in task_label_s.split(f'{class_separator}'):
                label_word_distribution_keywords_dict = label_word_distribution.get(task_label, dict())


                for keyword in keywords:
                    label_word_distribution_keywords_dict[keyword] = label_word_distribution_keywords_dict.get(keyword, 0) + 1

                label_word_distribution[task_label] = label_word_distribution_keywords_dict

                # """
                # {
                #     "task1":          # This is task_label_word
                #         {
                #             "label1": {"word1":185,...}, # This is label_word_dist
                #             "label2": {}
                #         }
                # }
                # """
            task_label_word_distribution[eachtask] = label_word_distribution

        processing_bar.progress((idx/num_rows))

    st.balloons()

    for taskname in task_label_word_distribution:
        for label in task_label_word_distribution[taskname]:
            label_word_distribution = task_label_word_distribution[taskname][label]
            task_label_word_distribution[taskname][label] = [k for (k,v) in sorted(label_word_distribution.items(), key=lambda x: x[1], reverse=True)]
            task_label_word_distribution[taskname][label] = task_label_word_distribution[taskname][label][:topkwords]

    return task_label_word_distribution


# Python program to illustrate the intersection 
# of two lists 
def intersection(list1, list2): 
    """
    Intersection of two list means we need to take all those elements which are common to both of the initial lists and store them into another list.
    By the use of this hybrid method the complexity of the program falls to O(n). This is an efficient way of doing the following program.
    Reference: https://www.geeksforgeeks.org/python-intersection-two-lists/#:~:text=Intersection%20of%20two%20list%20means,the%20Intersection%20of%20the%20lists.

    Args:
        list1 -> List of keywords of Class-1
        list2 -> List of keywords of Class-2
    """
    # Use of hybrid method 
    temp = set(list2) 
    list3 = [value for value in list1 if value in temp] 
    cooccurence_percent = (len(list3)/max(len(list1), len(list2)))*100
    cooccurence_percent = int(cooccurence_percent)
    return cooccurence_percent

def create_class_keywordmatrix(task_label_word_distribution: dict):
    """
    Reference: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    Reference: https://python-graph-gallery.com/92-control-color-in-seaborn-heatmaps/
    
    # plot using a color palette
    sns.heatmap(df, cmap="YlGnBu")
    sns.heatmap(df, cmap="Blues")
    sns.heatmap(df, cmap="BuPu") 
    sns.heatmap(df, cmap="Greens")

    Args: task_label_word_distribution
        {
            'task1':
                {
                    'label1': {"word1":185,...}, # This is label_word_dist
                    'label2': {"word3":1,...}

                },
            'task2':
                {
                    'label4': {"word1":185,...}
                }
        }
    """
    for taskname in task_label_word_distribution:
        labels = task_label_word_distribution[taskname]
        label_cocurrences = np.zeros((len(labels), len(labels)), dtype=np.int)
        for idx1,label1 in enumerate(labels):
            for idx2,label2 in enumerate(labels):
                list1 = task_label_word_distribution[taskname][label1]
                list2 = task_label_word_distribution[taskname][label2]
                cooccurence_percent = intersection(list1, list2)
                label_cocurrences[idx1][idx2] = cooccurence_percent
    
        df_cm = pd.DataFrame(label_cocurrences.tolist(), labels, labels)
        plt.figure(figsize=(25,25))
        sn.set(font_scale=1.8) # for label size
        sn.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 15}) # font size

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
    tasks_labels_distributions = create_class_distribution(df, taskcols=tasklabel_selected, class_separator=',')

# If user wants to look at the data, this will let them see it
keywords_preview = st.checkbox('Look at top _k_ words per category & task')
if keywords_preview:
    task_label_word_distribution = get_keywords_per_class(df=df, textcol=text_selected, taskcols= tasklabel_selected, class_separator=',', topkwords=500)
    st.pyplot(create_class_keywordmatrix(task_label_word_distribution))
    # st.write(task_label_word_distribution)

#  python -m streamlit run C:\Users\User\Downloads\TopicDetectorInDebate\mainTdetect.py


# https://plotly.com/python/discrete-color/
# https://github.com/MaartenGr/BERTopic

# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import random
import re

from io import StringIO


#from bertopic import BERTopic
#from bertopic.representation import KeyBERTInspired


def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")


def split_into_para(text):
  exmpl = text.split(". ")
  nn = len(exmpl)

  splited = []

  if nn % 5 == 0:
    n0 = 0
    n1 = 5

    for _ in range( int(nn/5) ):
      #print( exmpl[n0:n1] )
      para_split = exmpl[n0:n1]
      para_split = ". ".join(para_split)
      splited.append( para_split )
      n0+=5
      n1+=5

  elif nn % 4 == 0:
    n0 = 0
    n1 = 4

    for _ in range( int(nn/4) ):
      #print( exmpl[n0:n1] )
      para_split = exmpl[n0:n1]
      para_split = ". ".join(para_split)
      splited.append( para_split )
      n0+=4
      n1+=4

  else:
    n0 = 0
    n1 = 6

    for _ in range( int( np.ceil(nn/6)) ):
      #print( exmpl[n0:n1] )
      para_split = exmpl[n0:n1]
      para_split = ". ".join(para_split)
      splited.append( para_split )
      n0+=6
      n1+=6

  return splited



def read_file(file):
    with tempfile.NamedTemporaryFile(mode="wb") as temp:
        bytes_data = file.getvalue()
        temp.write(bytes_data)
        df = MyFunctionReadsFromPathAndAggregations(temp.name)
        return df


#  *********************** sidebar  *********************
with st.sidebar:
    #standard
    st.write("### Parameters")
    add_spacelines(2)
    st.write('Upload your debate in a **.txt** format')
    uploaded_file = st.file_uploader('', type = 'txt', accept_multiple_files = False, label_visibility = 'collapsed')
    if uploaded_file is not None:
        st.success('File uploaded correctly!')
    else:
        st.stop()


st.title('Topic detection in structured debates')


#with open(uploaded_file.name, 'r') as file:
#    data = uploaded_file.read()

data = uploaded_file.getvalue()
#st.write(data)

data=StringIO(uploaded_file.getvalue().decode('utf-8'))
data=data.read()

data_list = data.split("\r\n\r\n")

data = pd.DataFrame( {'text':data_list} )
data['speaker'] = data.text.apply(lambda x: x.split(":")[0] )
dd0 = data.speaker.value_counts().reset_index()
dd0 = dd0[dd0['count'] >= 5]
data = data[data.speaker.isin(dd0.speaker.unique())]

data['nwords'] = data.text.apply(lambda x: len( x.split() ) )
dd1 = data.groupby('speaker')['nwords'].sum().sort_values().reset_index()
dd1_politicians = dd1[dd1['nwords'] >= dd1['nwords'].mean()]['speaker'].tolist()
data['speaker_category'] = np.where( data.speaker.isin(dd1_politicians), 'politician', 'moderator' )

data['sentence'] = data.text.apply( lambda x: split_into_para(x[4:].strip()) )

data2 = data.copy()
data2 = data2.explode('sentence')
data2 = data2.reset_index()

st.write(data2)


# Fine-tune your topic representations
representation_model = KeyBERTInspired()
topic_model = BERTopic(representation_model=representation_model, min_topic_size = 15)

docs = data2['sentence']
topics, probs = topic_model.fit_transform(docs)
freq = topic_model.get_topic_info()

st.write(freq)
# https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-topics-over-time

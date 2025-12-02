import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
import matplotlib.pyplot as plt
import altair as alt
from brains import TranslateAndEmotion as TaEModel

st.title("Программная инженерия: лабораторная работа №3")
st.header("Выполнили Тихомиров Алексей и Рудин Валентин")
st.subheader("Приложение позволяет определить эмоциональную окраску отзывов о фильме")

colors = {
        'love': '#FF6B6B',
        'admiration': '#FFA500',
        'approval': '#32CD32',
        'neutral': '#87CEEB',
        'disappointment': '#6A5ACD',
        'disapproval': '#9370DB',
        'anger': '#DC143C',
        'disgust': '#8B4513'
    }

@st.cache_resource
def load_model():
    model = TaEModel()
    return model

model = load_model()

uploaded_file = st.file_uploader("Загрузите файл с разделителями", ".txt")

if uploaded_file:
    
    text = uploaded_file.getvalue().decode("utf-8")
    comments = model.get_sentences_from_text(text)
    translated_commets = model.translate_all_sentences(comments)
    model_outputs = model.classified_emotions_from_data(translated_commets)
        
    result, label_counts = model.count_emotions(model_outputs)

    bar_colors = [colors.get(category, '#888888') for category in label_counts.keys()]
    result["Цвет"] = bar_colors
    df = pd.DataFrame(result)
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="Эмоции",
            y="Количество",
            color="Цвет"
        )
    )
    st.altair_chart(bars, theme=None, use_container_width=True) 
    


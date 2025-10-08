import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter

st.title("Программная инженерия: лабораторная работа №3")
st.header("Выполнили Алексей Тихомиров и Рудин Валентин")
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
def load_models():
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    translator = pipeline("translation_ru_to_en", "Helsinki-NLP/opus-mt-ru-en")
    return (classifier, translator)

classifier, translator = load_models()

uploaded_file = st.file_uploader("Загрузите файл для перевода с разделителями \\n", ".txt")

if uploaded_file:
    
    text = uploaded_file.getvalue().decode("utf-8")
    sentences = text.split("\n")
    data = []
    for i, sentence in enumerate(sentences):
        text = translator(sentence)
        model_outputs = classifier(text[0]['translation_text'])
        data.append(model_outputs[0][0])

    # Подсчет частоты меток
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    result = {
        "Эмоции": [],
        "Количество": []
    }
    for key in label_counts.keys():
        result["Эмоции"].append(key)
        result["Эмоции"].append(label_counts[key])

    bar_colors = [colors.get(category, '#888888') for category in label_counts.keys()]
    st.bar_chart(result, x="Эмоции", y= "Количество", color=bar_colors, horizontal=False)    
    


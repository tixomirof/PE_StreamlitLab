import streamlit as st
import pandas as pd

st.title("Программная инженерия: лабораторная работа №3")
st.header("Вариант-11 (Тихомиров Алексей)")

st.write("Исходная таблица данных:")
df = pd.read_csv("https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv")
st.dataframe(df)

age_category = st.selectbox(
    "Выберите возрастную категорию:",
    ('Молодой', 'Среднего возраста', 'Старый')
)

if age_category == 'Молодой':
    filtered_df = df[df['Age'] < 30]
elif age_category == 'Среднего возраста':
    filtered_df = df[(df['Age'] >= 30) & (df['Age'] < 60)]
else:
    filtered_df = df[df['Age'] >= 60]

if not filtered_df.empty:
    survival_rate = filtered_df['Survived'].mean()
    st.write(f"Доля спасенных: {survival_rate:.2%}")
    st.write(f"Доля погибших: {1 - survival_rate:.2%}")
else:
    st.write("Нет данных для выбранной категории")

st.dataframe(filtered_df)
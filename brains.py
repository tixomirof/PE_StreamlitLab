import pandas as pd
from collections import Counter
from transformers import pipeline


class TranslateAndEmotion:

    def __init__(self):
        self.load_models()

    def load_models(self):
        self.classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        self.translator = pipeline("translation_ru_to_en", "Helsinki-NLP/opus-mt-ru-en")
        self.data_comments_file_path = "testfile.txt"

    def get_sentences_from_text(self, text):
        sentences = text.split("\n")
        return sentences

    def transalate_sentence(self, sentence):
        translated_sentence = self.translator(sentence)
        return translated_sentence[0]['translation_text']
    
    def translate_all_sentences(self, sentences):
        result = []
        for sentence in sentences:
            res = self.transalate_sentence(sentence)
            result.append(res)
        return result

    def set_new_comment_in_data(self, sentence):
        with open(self.data_comments_file_path, "a", encoding="utf-8") as f:
            f.write(f"{sentence}\n")

    def get_comments_data(self):
        comments_data = []
        with open(self.data_comments_file_path, "r", encoding="utf-8") as f:
            for line in f:
                comments_data.append(line)
        return comments_data
    
    def get_translated_comments_data(self):
        comments = self.get_comments_data()
        translated_data = []
        for comment in comments:
            res = self.transalate_sentence(comment)
            translated_data.append(res)
        return translated_data
    
    def classified_emotions_from_data(self, data):
        result = []
        for comment in data:
            model_outputs = self.classifier(comment)
            result.append(model_outputs[0][0])
        return result

    def classified_emotions_from_file(self):
        translated = self.get_translated_comments_data()
        result = self.classified_emotions_from_data(translated)
        return result
    
    def count_emotions(self, model_output_data):
        labels = [item['label'] for item in model_output_data]
        label_counts = Counter(labels)
        result = {
            "Эмоции": [],
            "Количество": []
        }
        for key in label_counts.keys():
            result["Эмоции"].append(key)
            result["Количество"].append(label_counts[key])
        return (result, label_counts)


import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Keep downloads for heroku
import nltk
# Below line was needed once while running on Windows 10 and Ubuntu
nltk.download('punkt')
# Below line was needed once while running on Ubuntu
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


class ChatBot:
    
    def __init__(self, bot_name, intents_file_path, all_data_pickle_file_path, chatbot_model_file_path):
        self.words, self.classes, self.training = self.load_pickle_file(all_data_pickle_file_path)
        self.model = load_model(chatbot_model_file_path)
        self.intents = json.loads(open(intents_file_path).read())
        self.bot_name = bot_name
    
    def load_pickle_file(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as file:
            words, classes, training = pickle.load(file)
        return words, classes, training

    def clean_sentence(self, sentence):
        lem = WordNetLemmatizer()
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lem.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        predictions = self.model.predict(np.array([bow]))[0]
        error_thresh = 0.25
        results = [[i, r] for i, r in enumerate(predictions) if r > error_thresh]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, msg):
        intents_list = self.predict_class(msg)
        tag = intents_list[0]['intent']
        prob = float(intents_list[0]['probability'])
        list_of_intents = self.intents['intents']
        error_thresh = 0.60
        if prob > error_thresh:
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
        # if the error_thresh is not met respond IDK
        else:
            # the first item in intents should be responses to unknown inputs
            result = random.choice(self.intents['intents'][0]['responses'])
        return result


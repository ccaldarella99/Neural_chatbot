# This script runs the chat bot on Flask

import datetime
import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import ChatBot

# Keep downloads for heroku
import nltk
# Below line was needed once while running on Windows 10 and Ubuntu
nltk.download('punkt')
# Below line was needed once while running on Ubuntu
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# import time
from flask import Flask, render_template
from flask_socketio import SocketIO

# Files and Paths
intents_file_path = '../data/intents.json'
all_data_pickle_file_path = '../models/all_data.pkl'
chatbot_model_file_path = '../models/chatbotmodel.h5'


def log_dt(log_level = 'INFO'):
    return datetime.datetime.now().isoformat() + ' [' + log_level.upper() + '] : '


app = Flask(__name__)
# change this and make it pull from a file so it does not pull to github
app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app, cors_allowed_origins="*") # Local
socketio = SocketIO(app, cors_allowed_origins="https://sam-grief-bot.herokuapp.com") # Heroku


@app.route('/', methods=['POST', 'GET'])
def sessions():
    return render_template('index.html') #, form=form, chat_responses=chat_responses)


def messageReceived(methods=['GET', 'POST']):
    print('message was recieved.')


def messageSent(methods=['GET', 'POST']):
    print('message was sent.')


@socketio.on('my event')
def handle_my_custom_event(json_msg, methods=['GET', 'POST']):
    print(f'{log_dt()} RCVD: {str(json_msg)}')
    socketio.emit('my response', json_msg, callback=messageReceived)
    
    print(f'\n\nTRY EXTRACT MESSAGE FROM: {json_msg}')
    get_msg = json_msg['message']
    send_msg = cb.get_response(get_msg)
    print(f'TRY EXTRACT MESSAGE : {send_msg}\n')
    mock_data = {'user_name': cb.bot_name.upper(), 'message': send_msg}

    sam_json = json.dumps(mock_data, sort_keys=True)
    socketio.emit('sam response', sam_json, callback=messageSent)
    print(f'{log_dt()} SENT: {str(sam_json)}')


if(__name__ == '__main__'):
    bot_name = 'Sam'
    cb = ChatBot.ChatBot(bot_name, intents_file_path,
                 all_data_pickle_file_path, chatbot_model_file_path)
    socketio.run(app, debug=True)

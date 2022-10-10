'''
 python -m client2
'''

import requests
import json
from flask import Flask, request
from flask_log import logger
import torch

logger = logger('Client 2')
server_url = 'http://127.0.0.1:5000/server-receive'
mq = [logger.get_str('Client start. ')]

app = Flask(__name__)


def add_one(lst):
    return [i + 1 for i in lst]


@app.route('/')
def home():
    html = '<h1>Client 2 Homepage</h1>'
    for m in mq:
        html = html + f'<p>{m}</p>'
    return html


@app.route('/client-receive', methods=['POST'])
def on_receive():
    if request.method == 'POST':
        json_data = request.get_json()
        data_list = json.loads(json_data)

        sender = data_list[0]
        model = data_list[1]
        remain_round = data_list[2]
        #app.logger.info("sender is %s",sender)
        mq.append(logger.get_str(f'Receive file from {sender}'))

        #update model
        updated_model = add_one(model)
        remain_round -= 1

        #send to server
        data_list = [1,updated_model,remain_round]
        json_data = json.dumps(data_list)
        requests.post(url=server_url, json=json_data)
        mq.append(logger.get_str(f'Send model from client 2'))

        return logger.get_str(f'Receive file from {sender}')


if __name__ == '__main__':
    app.run(port=5002, debug=True)

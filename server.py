'''
 flask --app flask_starter --debug run
 python -m flask_starter
'''

import requests
import json
from flask import Flask, redirect, url_for, request
from flask_log import logger
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import pickle

logger = logger('Server')
mq = [logger.get_str('Server start. ')]

num_of_client = 1
client_url = []
for i in range(num_of_client):
    client_url.append('http://127.0.0.1:500'+ str(i+1) + '/client-receive')
    #= ['http://127.0.0.1:5001/client-receive','http://127.0.0.1:5002/client-receive']
count = 0
model_list = [[] for _ in range(num_of_client)]

app = Flask(__name__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization


def FedAvg(models):
    n = len(models)
    sum = [0 for _ in range(10)]
    for model in models:
        for i in range(10):
            sum[i] = sum[i] + model[i]

    average_model = [i/n for i in sum]
    return average_model


@app.route('/')
def home():
    html = '<h1>Server Homepage</h1>'
    for m in mq:
        html = html + f'<p>{m}</p>'

    return html


@app.route('/start')
def start():
    model = CNN()
    mq.append(logger.get_str(f'Init model: {model}'))

    data_list = ['server',model,num_of_client]
    pickled_data = pickle.dumps(data_list)
    #data = {'sender': 'server', 'model': json_model, 'remain_round': iter_round}

    #send data
    for i in range(num_of_client):
        requests.post(url=client_url[i], data = pickled_data)
        mq.append(logger.get_str(f'Send model to {client_url[i]}'))

    return redirect(url_for('home'))


@app.route('/server-receive', methods=['POST'])
def on_receive():
    global count

    if request.method == 'POST':
        pickled_data = request.get_data()
        data_list = pickle.loads(pickled_data)
        sender_index = data_list[0]
        model_list[sender_index] = data_list[1]
        accuracy = data_list[2]
        count = count + 1
        mq.append(logger.get_str(f'Receive data from client : {sender_index} {accuracy}'))


#    if count == num_of_client:
#        model = FedAvg(model_list)
#        count = 0
#
#        if remain_round > 0:
#            mq.append(logger.get_str(f'Remain round: {remain_round}'))
#
#            for i in range(num_of_client):
#                data_list = ['server',model,remain_round]
#                json_data = json.dumps(data_list)
#                requests.post(url=client_url[i], json=json_data)
#                mq.append(logger.get_str(f'Send to {client_url[i]} with model {model}'))

    return logger.get_str(f'Receive file')


if __name__ == '__main__':
    app.run(port=5000, debug=True)

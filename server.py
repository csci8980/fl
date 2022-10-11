'''
 flask --app flask_starter --debug run
 python -m flask_starter
'''

import requests
from flask import Flask, redirect, url_for, request
from flask_log import logger
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import pickle

logger = logger('Server')
mq = [logger.get_str('Server start. ')]

#client numbers and URL
num_of_client = 5
client_url = []
for i in range(num_of_client):
    client_url.append('http://127.0.0.1:500'+ str(i+1) + '/client-receive')
#ML model parameters
count = 0
model_list = [[] for _ in range(num_of_client)]
accuracy_list = [0 for _ in range(num_of_client)]
remain_epoch = 10
desired_accuracy = 0.98

app = Flask(__name__)

#ML model abstraction
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

#FedAvg algorithm
def FedAvg(models):
    mq.append(logger.get_str(f'Performing FedAvg for epoch: {10 - remain_epoch}'))
    n = len(models)
    avg_model = models[0]
    sd_avg_model = avg_model.state_dict()

    for i in range(1,n):
        sd_model = models[i].state_dict()
        for key in sd_model:
            sd_avg_model[key] = sd_avg_model[key] + sd_model[key]

    for key in sd_avg_model:
        sd_avg_model[key] = sd_avg_model[key]/n

    new_model = CNN()
    new_model.load_state_dict(sd_avg_model)
    return new_model

#home page
@app.route('/')
def home():
    html = '<h1>Server Homepage</h1>'
    for m in mq:
        html = html + f'<p>{m}</p>'

    return html

#page for ML program to start
@app.route('/start')
def start():
    #create ML model
    model = CNN()
    mq.append(logger.get_str(f'Init a ML model: {model}'))

    #wrap data with pickle
    data_list = ['server',model,num_of_client]
    pickled_data = pickle.dumps(data_list)

    #send data
    for i in range(num_of_client):
        mq.append(logger.get_str(f'Send model to {client_url[i]}'))
        requests.post(url=client_url[i], data = pickled_data)

    return redirect(url_for('home'))

#page to receive ML models from clients and send new averaged model
@app.route('/server-receive', methods=['POST'])
def on_receive():
    global count, remain_epoch
    #receive updated ML models from clients
    if request.method == 'POST':

        pickled_data = request.get_data()
        data_list = pickle.loads(pickled_data)
        sender_index = data_list[0]
        model_list[sender_index] = data_list[1]
        accuracy_list[sender_index] = data_list[2]
        count = count + 1
        mq.append(logger.get_str(f'Receive data from client : {sender_index}'))

    #check if all models from clients return to server
    if count == num_of_client:
        #do FedAvg
        new_model = FedAvg(model_list)
        current_accuracy = sum(accuracy_list)/len(accuracy_list)
        count = 0
        remain_epoch -= 1

        #send new models
        if remain_epoch > 0 and current_accuracy < desired_accuracy:
            mq.append(logger.get_str(f'Remaining global epoch: {remain_epoch}'))

            for i in range(num_of_client):
                data_list = ['server',new_model,num_of_client]
                pickled_data = pickle.dumps(data_list)
                mq.append(logger.get_str(f'Send to {client_url[i]} with new model'))
                requests.post(url=client_url[i], data=pickled_data)

        else:
            mq.append(logger.get_str(f'Federated Learning is finished with {10 - remain_epoch} epochs and accuracy is {current_accuracy}'))

    return ('',204)


if __name__ == '__main__':
    app.run(port=5000, debug=True)

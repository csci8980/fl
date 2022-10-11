'''
 flask --app flask_starter --debug run
 python -m flask_starter
'''

import requests
from flask import Flask, request
from flask_log import logger
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import pickle

logger = logger('Client 17')
mq = [logger.get_str('Client start. ')]

server_url = 'http://127.0.0.1:5000/server-receive'

app = Flask(__name__)

#load data for this client
train_data = datasets.MNIST(root = 'data',train = True,transform = ToTensor(),download = True,)
test_data = datasets.MNIST(root = 'data',train = False,transform = ToTensor())
train_data1, train_data2 = torch.utils.data.random_split(train_data, [12000,48000])
test_data1, test_data2 = torch.utils.data.random_split(test_data, [5000,5000])
train_data = train_data1
test_data = test_data1
del train_data2


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
        return output, x


#ML train function
def train(num_epochs, cnn, loaders):
    mq.append(logger.get_str(f'Local training starts'))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

    cnn.train()
    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i+1) % 100 == 0:
                mq.append(logger.get_str(f'Local epoch is {epoch+1}. Loss is {loss.item()}'))
                #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
        pass
    pass


#ML test function
def test(model,loaders):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
        return accuracy


#load data, train and test model
def update_model(model):
    loaders = {'train' : torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True,num_workers=1),
                'test'  : torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=True,num_workers=1),}

    num_epochs = 2
    train(num_epochs, model, loaders)
    accuracy = test(model,loaders)
    return model,accuracy

#home page
@app.route('/')
def home():
    html = '<h1>Client 1 Homepage</h1>'
    for m in mq:
        html = html + f'<p>{m}</p>'
    return html

#page for client to receive data and send data
@app.route('/client-receive', methods=['POST'])
def on_receive():
    if request.method == 'POST':
        #receive data from server
        pickled_data = request.get_data()
        data_list = pickle.loads(pickled_data)
        sender = data_list[0]
        model = data_list[1]
        num_of_client = data_list[2]
        mq.append(logger.get_str(f'Receive model from {sender}'))

        #update model
        updated_model,accuracy = update_model(model)

        #send to server
        data_list = [16,updated_model,accuracy]
        pickled_data = pickle.dumps(data_list)
        mq.append(logger.get_str(f'Send model from cilent 17'))
        requests.post(url=server_url, data=pickled_data)

        return ('',204)


if __name__ == '__main__':
    app.run(port=5017, debug=True)

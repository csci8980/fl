"""
    python -m client 5001
"""

import argparse
import configparser
import pickle
import random
import time

import requests
from flask import Flask, request, redirect, url_for, jsonify
from torchvision.transforms import ToTensor

from CNN import CNN, update_model
from SampledMNIST import SampledMNIST
from logger import Logger


class Client:
    id = None
    host = None
    port = None
    url = None

    def __init__(self, _id_, host, port):
        self.id = _id_
        self.host = host
        self.port = port
        self.url = host + f':{port}/client-receive'


def random_sleep(prob, sleep_time):
    if random.random() < prob:
        mq.append(logger.get_str(f"Sleep {sleep_time} seconds"))
        time.sleep(sleep_time)
        mq.append(logger.get_str("Wake up"))


app = Flask(__name__)


# home page
@app.route('/')
def home():
    html = '<h1>Client Homepage</h1>'
    html += f'''
    <button onclick="window.location.href='{client_host}:{client_port}/register';">
        Register
    </button>
    '''
    for m in mq:
        html += f'<p>{m}</p>'
    return html


@app.route('/register')
def register():
    url_params = {'client_host': client_host,
                  'client_port': client_port,
                  'train_count': train_count,
                  'test_count': test_count,
                  'skew_label': skew_label,
                  'skew_prop': skew_prop}
    r = requests.post(url=f'http://127.0.0.1:5000/register/{client_port}', params=url_params)  # register client
    if r.status_code == 200:
        mq.append(logger.get_str('Successfully register client'))
    else:
        RuntimeError('Fail register client')

    return redirect(url_for('home'))


# page for client to receive data and send data
@app.route('/client-receive', methods=['POST'])
def on_receive():
    if request.method == 'POST':
        # receive data from server
        pickled_model = request.get_data()
        model = pickle.loads(pickled_model)
        assert isinstance(model, CNN)
        curr_epoch = int(request.args.get('curr_epoch'))
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Receive model from server'))

        # random sleep
        random_sleep(client_sleep_prob, client_sleep_second)

        # update model
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Start client training'))
        updated_model, accuracy = update_model(model, train_data, test_data)
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Done client training'))

        # send to server
        pickled_model = pickle.dumps(updated_model)
        url_params = {'client_port': client_port, 'curr_epoch': curr_epoch, 'accuracy': accuracy}
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Send model to server'))
        r = requests.post(url=server_url, data=pickled_model, params=url_params)
        if r.status_code == 200:
            return jsonify(url_params), 200
        else:
            print('\n !!! Client: Something wrong when client send model to server !!! \n')
            return '', 204


if __name__ == '__main__':
    # read argument
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    args = parser.parse_args()
    client_port = int(args.port)

    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')

    # read server config
    server_host = config['server']['host']
    server_port = int(config['server']['port'])
    server_url = server_host + f":{server_port}/server-receive/{client_port}"

    # read client config
    client_host = config['client']['host']
    client_sleep_second = int(config['timeout']['client_sleep_second'])
    client_sleep_prob = float(config['timeout']['client_sleep_prob'])

    # load data
    train_count = int(config['data']['train_count'])
    test_count = int(config['data']['test_count'])
    skew_client_port = int(config['data']['skew_client_port'])
    skew_label = int(config['data']['skew_label'])
    skew_prop = int(config['data']['skew_prop'])
    if client_port != skew_client_port:
        skew_label = -1
        skew_prop = -1
        train_data = SampledMNIST(root='data', train=True, transform=ToTensor(), download=True, n_total=train_count)
        test_data = SampledMNIST(root='data', train=False, transform=ToTensor(), download=True, n_total=test_count)
    else:
        train_data = SampledMNIST(root='data', train=True, transform=ToTensor(), download=True, n_total=train_count, skew_label=skew_label, skew_prop=skew_prop)
        test_data = SampledMNIST(root='data', train=False, transform=ToTensor(), download=True, n_total=test_count, skew_label=skew_label, skew_prop=skew_prop)

    # init logger and dashboard mq
    logger = Logger(f'Client {client_port}')
    mq = [logger.get_str('Client start.')]

    # start client
    app.run(port=client_port, debug=True)

"""
    python -m client 5001
"""

import argparse
import configparser
import json
import pickle

import requests
from flask import Flask, request, redirect, url_for, jsonify

from CNN import CNN, update_model
from SampledEMNIST import SampledEMNIST
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
                  'label_dist': dist,
                  'data_count': count}
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
        model_name = request.args.get('model_name')
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Receive model from server'))

        # update model
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Start client training'))
        updated_model, accuracy, tau = update_model(model, train_data, test_data, model_name)
        # print("Here updated model is", updated_model)
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Done client training'))

        # send to server
        pickled_model = pickle.dumps(updated_model)
        url_params = {'client_port': client_port, 'curr_epoch': curr_epoch, 'accuracy': accuracy, 'train_count': train_count, 'tau': tau}
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Send model to server'))
        r = requests.post(url=server_url, data=pickled_model, params=url_params)
        if r.status_code == 200:
            return jsonify(url_params), 200
        else:
            print('\n !!! Client: Something wrong when client send model to server !!! \n')
            return '', 204


def get_data_profile(port, data_profile_dict):
    if str(port) in data_profile_dict:
        profile = data_profile_dict[str(port)].split('_')
        dist = profile[0]
        count = profile[1]
        return dist, count
    else:
        # default 'even_more'
        return 'even', 'more'


def load_data(port, dist, count):
    if count == 'more':
        train_count = 6000
        test_count = 1000
    else:
        train_count = 600
        test_count = 100
    num = port - 5001
    train_data_name = f'train_{dist}_{train_count}_{num}.pkl'
    test_data_name = f'test_{dist}_{test_count}_{num}.pkl'
    prefix = 'data/SampledEMNIST/pickle/'
    with open(prefix + train_data_name, 'rb') as file:
        train_data = pickle.load(file)
        assert isinstance(train_data, SampledEMNIST)
    with open(prefix + test_data_name, 'rb') as file:
        test_data = pickle.load(file)
        assert isinstance(test_data, SampledEMNIST)
    return train_data, test_data


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

    # load data
    profile_dict = json.loads(config['data']['profile'].replace('\n', ''))
    dist, count = get_data_profile(client_port, profile_dict)
    train_data, test_data = load_data(client_port, dist, count)
    train_count = len(train_data)
    test_count = len(test_data)

    # init logger and dashboard mq
    logger = Logger(f'Client {client_port}')
    mq = [logger.get_str('Client start.')]

    # start client
    app.run(port=client_port, debug=True)

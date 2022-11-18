"""
    python -m server
"""
import concurrent.futures
import configparser
import pickle
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np
import pandas as pd
import requests
from flask import Flask, redirect, request, url_for

from CNN import CNN
from FedAvg import fed_avg
from client import Client
from logger import Logger

app = Flask(__name__)


# home page
@app.route('/')
def home():
    html = '<h1>Server Homepage</h1>'
    html += f'''
    <button onclick="window.location.href='http://127.0.0.1:{server_port}/start';"> 
        Start
    </button>
    '''
    html += f'<h2>Stop accuracy: {desired_accuracy}</h2>'
    html += f'<h2>Maximum epoch: {total_epoch}</h2>'
    html += f'<h2>Server timeout second: {server_timeout_second}</h2>'
    html += f'<h2>Client sleep probability: {client_sleep_prob}</h2>'
    html += f'<h2>Client sleep second: {client_sleep_second}</h2>'
    html += f'<h2>Minimum quorum requirement: {min_quorum_requirement}</h2>'
    html += f'<h2>Current client count: {len(client_dict)}, ID (port):[{", ".join(map(str, client_dict))}]</h2>'
    html += dashboard.to_html()

    for m in mq:
        html += f'<p>{m}</p>'

    return html


# client register page
@app.route('/register/<client_port>', methods=['POST'])
def register(client_port):
    client_port = int(client_port)
    if request.method == 'POST':
        if client_port not in client_dict:
            client_host = request.args.get('client_host')
            client_port = int(request.args.get('client_port'))
            client = Client(_id_=client_port, host=client_host, port=client_port)
            client_dict[client.id] = client
            mq.append(logger.get_str(f'Register client {client.id} with url: {client.url}'))
            dashboard.loc[client.id] = np.nan
            # client info
            dashboard.loc[client.id, 'train_count'] = request.args.get('train_count')
            dashboard.loc[client.id, 'test_count'] = request.args.get('test_count')
            dashboard.loc[client.id, 'skew_label'] = request.args.get('skew_label')
            dashboard.loc[client.id, 'skew_prop'] = request.args.get('skew_prop')

    return redirect(url_for('home'))


# send model to client function executed by each thread
def thread_send_model(cid, curr_epoch, pickled_model):
    client = client_dict[cid]
    mq.append(logger.get_str(f'Epoch {curr_epoch}: Send model to {cid}'))
    url_params = {'curr_epoch': curr_epoch}
    return requests.post(url=client.url, data=pickled_model, params=url_params)


# broadcast model and wait for client response in a multi-thread way
def broadcast_model(model, curr_epoch):
    pickled_model = pickle.dumps(model)
    with ThreadPoolExecutor(max_workers=6) as thread_pool:
        futures = [thread_pool.submit(thread_send_model, c, curr_epoch, pickled_model) for c in client_dict]

        try:
            # iterate future objects with a timeout on the first task completing
            for future in as_completed(futures, timeout=server_timeout_second):  # a future object will be returned once a thread job finish
                response = future.result()
                assert isinstance(response, requests.Response)
                if response.status_code == 200:
                    client_return_json = response.json()
                    port = client_return_json['client_port']
                    accuracy = client_return_json['accuracy']
                    signup_dict[curr_epoch].add(port)
                    dashboard.at[port, curr_epoch] = accuracy
                    mq.append(logger.get_str(f'Epoch {curr_epoch}: Receive in-time response from {port}'))
                    # check if reach the minimum requirement for quorum, if so, break
                    if len(signup_dict[curr_epoch]) >= min_quorum_requirement:
                        print(f'\nEpoch {curr_epoch}: Reach minimum quorum requirement before timeout. Break.\n')
                        mq.append(logger.get_str(f'Epoch {curr_epoch}: Reach minimum quorum requirement before timeout.'))
                        break

        except concurrent.futures.TimeoutError:  # raise when timeout exceed
            if len(signup_dict[curr_epoch]) < min_quorum_requirement:
                print(f'\nEpoch {curr_epoch}: Not enough quorum before timeout ({len(signup_dict[curr_epoch])}/{min_quorum_requirement})\n')
                mq.append(logger.get_str(f'Epoch {curr_epoch}: Not enough quorum before timeout ({len(signup_dict[curr_epoch])}/{min_quorum_requirement})'))
                return False
            else:
                print(f'\nEpoch {curr_epoch}: Receive {len(signup_dict[curr_epoch])} before timeout \n')

    mq.append(logger.get_str(f'Epoch {curr_epoch}: Receive {len(signup_dict[curr_epoch])} in-time response'))
    return True


# starting point of entire FL process
@app.route('/start')
def start():
    model = CNN()
    mq.append(logger.get_str(f'Init a ML model: {model}'))
    curr_epoch = 0
    curr_min_accuracy = -1
    while curr_epoch < total_epoch:
        print(f'\n Current epoch {curr_epoch}/{total_epoch} \n')
        mq.append(logger.get_str(f'Epoch {curr_epoch}: Start epoch {curr_epoch}/{total_epoch}'))
        is_broadcast_success = broadcast_model(model, curr_epoch)
        if not is_broadcast_success:  # break and abort FL if not enough quorums received from broadcasting
            mq.append(logger.get_str(f'Federated learning aborts at epoch {curr_epoch} for not enough quorum'))
            break
        curr_min_accuracy = dashboard[curr_epoch].min()
        mq.append(logger.get_str(f'Epoch {curr_epoch}: current accuracy {curr_min_accuracy}'))
        if curr_min_accuracy >= desired_accuracy:  # break if reach desired accuracy
            break
        else:
            in_time_client = signup_dict[curr_epoch]
            in_time_model = [model_dict[curr_epoch][c] for c in in_time_client]
            mq.append(logger.get_str(f'Epoch {curr_epoch}: Do FedAvg with {len(in_time_model)} models'))
            model = fed_avg(in_time_model)
            curr_epoch += 1

    mq.append(logger.get_str(f'Federated learning ends after {curr_epoch} epochs with accuracy {curr_min_accuracy}'))
    return redirect(url_for('home'))


# page to receive ML models from clients
# late models will also be stored, but they will not be used since they didn't sign up in time
@app.route('/server-receive/<port>', methods=['POST'])
def on_receive(port):
    if request.method == 'POST':
        client_port = int(port)
        client_epoch = int(request.args.get('curr_epoch'))
        pickled_model = request.get_data()
        model = pickle.loads(pickled_model)
        assert isinstance(model, CNN)
        model_dict[client_epoch][client_port] = model
        mq.append(logger.get_str(f'Epoch {client_epoch}: Receive data from client : {client_port}'))

    return 'Returned from server on_receive'


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    # read server config
    server_port = int(config['server']['port'])

    # read ML config
    total_epoch = int(config['ml']['epoch'])
    desired_accuracy = float(config['ml']['accuracy'])

    # read timeout config
    server_timeout_second = int(config['timeout']['server_timeout_second'])
    client_sleep_second = int(config['timeout']['client_sleep_second'])
    client_sleep_prob = float(config['timeout']['client_sleep_prob'])

    # read quorum confid
    min_quorum_requirement = int(config['quorum']['min_quorum_requirement'])

    # init client model cache
    model_dict = {i: {} for i in range(total_epoch)}
    signup_dict = {i: set() for i in range(total_epoch)}  # in-time client response

    # init client registration
    client_dict = {}

    # init logger, mq and dashboard
    logger = Logger('Server')
    mq = [logger.get_str('Server start.')]
    client_info = ['train_count', 'test_count', 'skew_label', 'skew_prop']
    dashboard = pd.DataFrame(columns=client_info + list(range(total_epoch)))

    # start server
    app.run(port=server_port, debug=True)

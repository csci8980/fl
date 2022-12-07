"""
    python -m server
"""
import configparser
import pickle
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from flask import Flask, redirect, request, url_for

from CNN import CNN
from FedAlgorithm import fed_avg, fed_nova
from client import Client
from logger import Logger

app = Flask(__name__)


def save_data_and_fig(df):
    current_time = datetime.now().strftime('%H_%M_%S')
    title = f'{model_name}_{current_time}'
    csv_name = f'{title}.csv'
    df.to_csv(csv_name, index_label='port')

    # # plot
    # df = df.reset_index(names='port')
    # epoch = df.shape[1] - 5
    # client = len(df)
    # for c in range(client):
    #     c_label = f'{df.at[c, "label_dist"]}_{df.at[c, "data_count"]}'
    #     if df.at[c, "label_dist"] == 'zipf':
    #         plt.plot(range(epoch), df.loc[c, [i for i in range(epoch)]], 'x-r', alpha=1, label=c_label)
    #     else:
    #         plt.plot(range(epoch), df.loc[c, [i for i in range(epoch)]], '.-', alpha=0.6, label=c_label)
    #
    # plt.grid(alpha=0.3)
    # plt.legend(loc='lower right', ncols=2)
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.xticks([i for i in range(epoch)])
    # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.title(model_name)
    # plt.savefig(f'{title}.png', dpi=360)


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
    html += f'<h2>Fed algo: {model_name}</h2>'
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
            dashboard.loc[client.id, 'label_dist'] = request.args.get('label_dist')
            dashboard.loc[client.id, 'data_count'] = request.args.get('data_count')

    return redirect(url_for('home'))


# send model to client function executed by each thread
def thread_send_model(cid, curr_epoch, pickled_model):
    client = client_dict[cid]
    mq.append(logger.get_str(f'Epoch {curr_epoch}: Send model to {cid}'))
    url_params = {'curr_epoch': curr_epoch, 'model_name': model_name}
    # print("model name is", model_name)
    return requests.post(url=client.url, data=pickled_model, params=url_params)


# broadcast model and wait for client response in a multi-thread way
def broadcast_model(model, curr_epoch):
    pickled_model = pickle.dumps(model)
    with ThreadPoolExecutor(max_workers=10) as thread_pool:
        futures = [thread_pool.submit(thread_send_model, c, curr_epoch, pickled_model) for c in client_dict]

        # iterate future objects
        for future in as_completed(futures):  # a future object will be returned once a thread job finish
            response = future.result()
            assert isinstance(response, requests.Response)
            if response.status_code == 200:
                client_return_json = response.json()
                port = client_return_json['client_port']
                accuracy = client_return_json['accuracy']
                dashboard.at[port, curr_epoch] = accuracy
                mq.append(logger.get_str(f'Epoch {curr_epoch}: Receive response from {port}'))


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
        broadcast_model(model, curr_epoch)
        curr_min_accuracy = dashboard[curr_epoch].min()
        mq.append(logger.get_str(f'Epoch {curr_epoch}: current accuracy {curr_min_accuracy}'))
        if curr_min_accuracy >= desired_accuracy:  # break if reach desired accuracy
            break
        else:
            to_fed_model = list(model_dict[curr_epoch].values())
            mq.append(logger.get_str(f'Epoch {curr_epoch}: Do FedLearning with {len(to_fed_model)} models'))
            # print("current model is ", to_fed_model)
            if model_name == "FedNova" or model_name == "FedMix":
                print(f'Epoch {curr_epoch}: do {model_name}')
                model = fed_nova(to_fed_model)
            else:
                print(f'Epoch {curr_epoch}: do FedAvg')
                model = fed_avg(to_fed_model)
            curr_epoch += 1

    mq.append(logger.get_str(f'Federated learning ends after {curr_epoch} epochs with accuracy {curr_min_accuracy}'))

    save_data_and_fig(dashboard)

    return redirect(url_for('home'))


# page to receive ML models from clients
@app.route('/server-receive/<port>', methods=['POST'])
def on_receive(port):
    if request.method == 'POST':
        client_port = int(port)
        client_epoch = int(request.args.get('curr_epoch'))
        client_train_count = int(request.args.get('train_count'))
        client_tau = int(request.args.get('tau'))
        # print('tau in sever is', client_tau)
        pickled_model = request.get_data()
        model = pickle.loads(pickled_model)
        assert isinstance(model, CNN)
        model_dict[client_epoch][client_port] = {'model': model, 'count': client_train_count, 'tau': client_tau}
        mq.append(logger.get_str(f'Epoch {client_epoch}: Receive data from client : {client_port}'))

    return 'Returned from server on_receive'


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    # read server config
    server_port = int(config['server']['port'])

    # read ML config
    total_epoch = int(config['ml']['epoch'])
    model_name = config['ml']['model_name']
    desired_accuracy = float(config['ml']['accuracy'])

    # init client model cache
    model_dict = {i: {} for i in range(total_epoch)}

    # init client registration
    client_dict = {}

    # init logger, mq and dashboard
    logger = Logger('Server')
    mq = [logger.get_str('Server start.')]
    client_info = ['train_count', 'test_count', 'label_dist', 'data_count']
    dashboard = pd.DataFrame(columns=client_info + list(range(total_epoch)))

    # start server
    app.run(port=server_port, debug=True)

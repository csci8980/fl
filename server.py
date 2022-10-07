'''
 flask --app flask_starter --debug run
 python -m flask_starter
'''

import requests
from flask import Flask, redirect, url_for, request

from flask_log import logger
from utils import write_to_pickle, read_from_pickle, gen_file_name

logger = logger('Server')

client_1_url = 'http://127.0.0.1:5001/client-receive'
mq = [logger.get_str('Server start. ')]
iter_round = 10

app = Flask(__name__)


def init_model():
    model = list(range(10))
    filename = gen_file_name('server')
    write_to_pickle(model, filename)
    mq.append(logger.get_str(f'Init model: {model}'))
    return filename


@app.route('/')
def home():
    html = '<h1>Server Homepage</h1>'
    for m in mq:
        html = html + f'<p>{m}</p>'
    return html


@app.route('/start')
def start():
    filename = init_model()
    data = {'sender': 'server', 'filename': filename, 'remain_round': iter_round}
    mq.append(logger.get_str(f'Send {filename} to {client_1_url}'))
    requests.post(url=client_1_url, json=data)
    return redirect(url_for('home'))


@app.route('/server-receive', methods=['POST'])
def on_receive():
    if request.method == 'POST':
        sender = request.json['sender']
        filename = request.json['filename']
        remain_round = request.json['remain_round']
        update = read_from_pickle(filename)
        mq.append(logger.get_str(f'Receive file from {sender}: {filename}'))
        mq.append(logger.get_str(f'Read value: {update}'))
        if remain_round > 0:
            mq.append(logger.get_str(f'Remain round: {remain_round}'))
            #redirect(url_for('next', filename=filename))
            data = {'sender': 'server', 'filename': filename,'remain_round': remain_round}
            mq.append(logger.get_str(f'Send {filename} to {client_1_url}'))
            requests.post(url=client_1_url, json=data)

        return logger.get_str(f'Receive file from {sender}: {filename}')


@app.route('/next/<filename>')
def next(filename):
    data = {'sender': 'server', 'filename': filename}
    mq.append(logger.get_str(f'Send {filename} to {client_1_url}'))
    requests.post(url=client_1_url, json=data)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(port=5000, debug=True)

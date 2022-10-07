'''
 flask --app flask_starter --debug run
 python -m flask_starter
'''

import requests
from flask import Flask, request

from flask_log import logger
from utils import write_to_pickle, read_from_pickle, gen_file_name

logger = logger('Client 1')
server_url = 'http://127.0.0.1:5000/server-receive'
mq = [logger.get_str('Client start. ')]

app = Flask(__name__)


def double(lst):
    return [i * 2 for i in lst]


@app.route('/')
def home():
    html = '<h1>Client 1 Homepage</h1>'
    for m in mq:
        html = html + f'<p>{m}</p>'
    return html


@app.route('/client-receive', methods=['POST'])
def on_receive():
    if request.method == 'POST':
        sender = request.json['sender']
        filename = request.json['filename']
        remain_round = request.json['remain_round']
        #app.logger.info("sender is %s",sender)

        lst = read_from_pickle(filename)
        mq.append(logger.get_str(f'Receive file from {sender}: {filename}'))
        mq.append(logger.get_str(f'Read value: {lst}'))

        update = double(lst)
        filename = gen_file_name('client_1')
        write_to_pickle(update, filename)
        remain_round -= 1

        data = {'sender': 'server', 'filename': filename, 'remain_round': remain_round}
        requests.post(url=server_url, json=data)

        return logger.get_str(f'Receive file from {sender}: {filename}')


if __name__ == '__main__':
    app.run(port=5001, debug=True)

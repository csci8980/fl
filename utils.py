import pickle
from datetime import datetime


def write_to_pickle(obj, file):
    with open(file, 'wb') as fp:
        pickle.dump(obj, fp)


def read_from_pickle(file):
    with open(file, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def gen_file_name(server_or_client):
    if server_or_client == 'server' or server_or_client.startswith('client'):
        return f'{server_or_client}_{datetime.timestamp(datetime.now())}.pkl'
    else:
        raise ValueError('Should only be "server" or "client_x"')

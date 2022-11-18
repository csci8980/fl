"""
    Logger
"""

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s]-[%(levelname)s]: %(message)s',
    datefmt='%H:%M:%S',
)


class Logger:
    def __init__(self, role):
        self.role = role

    def get_str(self, msg):
        now = datetime.now()
        datetime_str = now.strftime("%H:%M:%S")
        return f"[{datetime_str}]-[{self.role}]: {msg}"

    def log(self, msg):
        print(self.get_str(msg))

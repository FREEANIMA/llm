import os
import queue
import threading

# app/core/globals.py 기준으로 두 단계 위 = app 폴더
BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_ROOT, "data")

from ..core.config import Config

config = Config()
console = queue.Queue()
stop_event = threading.Event()
BATCH_Q = queue.Queue(maxsize=256)
STOP_EVENT = threading.Event()
MONITOR = {

}
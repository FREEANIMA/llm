# 코랩 셀 3: 학습 스레드 시작
import threading
from app.train.run import Train
from app.core.globals import MONITOR

base_root = "/content/drive/MyDrive/model/llm/app"

trainer = Train(base_root=base_root)

t_loader = threading.Thread(target=trainer.batch_loader, daemon=True)
t_train = threading.Thread(target=trainer.train, daemon=True)
t_loader.start()
t_train.start()
import threading
import time
from IPython.display import display, clear_output, HTML
from app.train.run import Train
from app.core.globals import MONITOR

base_root = "/content/drive/MyDrive/model/llm/app"

trainer = Train(base_root=base_root)

t_loader = threading.Thread(target=trainer.batch_loader, daemon=True)
t_train = threading.Thread(target=trainer.train, daemon=True)
t_loader.start()
t_train.start()

# ---- 대시보드 ----
dashboard = display(HTML(""), display_id=True)

def render():
    s = MONITOR.get("stats") or {}
    infer = MONITOR.get("infer") or []

    loss  = s.get("loss")
    gnorm = s.get("gnorm")
    steps = s.get("steps", 0)
    lr    = s.get("lr")

    def fmt(v, f=".4f"):
        return format(v, f) if v is not None else "—"

    sample = s.get("sample_tags") or []
    sample_str = ", ".join(sample[0][:10]) if sample else "—"
    infer_str = ", ".join(infer[:15]) if infer else "—"

    html = f"""
    <div style="font-family: monospace; font-size: 14px; line-height: 1.6;">
      <b>steps</b> : {steps}<br>
      <b>loss</b>  : {fmt(loss)}<br>
      <b>gnorm</b> : {fmt(gnorm, '.3f')}<br>
      <b>lr</b>    : {lr if lr is not None else '—'}<br>
      <b>sample</b>: {sample_str}<br>
      <b>infer</b> : {infer_str}
    </div>
    """
    dashboard.update(HTML(html))

while t_train.is_alive():
    render()
    time.sleep(2)

render()   # 마지막 상태
print("training stopped")
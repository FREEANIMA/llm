import os
import threading
import time
import html as html_lib  # 변수명 html과 충돌 방지를 위해 별칭 사용

from app.train.run import Train
from app.core.globals import MONITOR

try:
    import google.colab  # noqa
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def get_base_root():
    if IN_COLAB:
        return "/content/drive/MyDrive/model/llm/app"
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    return os.path.join(here, "app")

base_root = get_base_root()
print("base_root:", base_root)

trainer = Train(base_root=base_root)

t_loader = threading.Thread(target=trainer.batch_loader, daemon=True)
t_train = threading.Thread(target=trainer.train, daemon=True)
t_loader.start()
t_train.start()

# ---- 대시보드 ----
if IN_COLAB:
    from IPython.display import display, HTML
    dashboard = display(HTML(""), display_id=True)

    def esc(tags):
        # 태그 리스트를 이스케이프해서 ", "로 합침
        return html_lib.escape(", ".join(str(t) for t in tags))

    def render():
        s = MONITOR.get("stats") or {}
        infer = MONITOR.get("infer") or {}

        def fmt(v, f=".4f"):
            return format(v, f) if v is not None else "—"

        sample = s.get("sample_tags") or []

        # infer 는 {"tags": [...], "ids": [...], "updated_step": N} dict
        if isinstance(infer, dict):
            infer_tags = infer.get("tags") or []
            infer_step = infer.get("updated_step", "—")
        else:
            infer_tags = list(infer)
            infer_step = "—"

        sample_str = esc(sample[0][:10]) if sample else "—"
        infer_str = esc(infer_tags[:15]) if infer_tags else "—"

        html_str = f"""
        <div style="font-family: monospace; font-size: 14px; line-height: 1.6;">
          <b>steps</b> : {s.get('steps', 0)}<br>
          <b>loss</b>  : {fmt(s.get('loss'))}<br>
          <b>gnorm</b> : {fmt(s.get('gnorm'), '.3f')}<br>
          <b>lr</b>    : {s.get('lr', '—')}<br>
          <b>sample</b>: {sample_str}<br>
          <b>infer</b> (step {infer_step}): {infer_str}
        </div>
        """
        dashboard.update(HTML(html_str))

    while t_train.is_alive():
        render()
        time.sleep(2)
    render()

else:
    # 로컬: 터미널에 주기 출력
    last = -1
    while t_train.is_alive():
        s = MONITOR.get("stats") or {}
        steps = s.get("steps", 0)
        if steps != last:
            loss = s.get("loss")
            gnorm = s.get("gnorm")
            lr = s.get("lr")
            loss_s = f"{loss:.4f}" if loss is not None else "—"
            gnorm_s = f"{gnorm:.3f}" if gnorm is not None else "—"
            lr_s = f"{lr:.1e}" if lr is not None else "—"
            print(f"[step {steps}] loss={loss_s} gnorm={gnorm_s} lr={lr_s}", end="\r", flush=True)
            last = steps
        time.sleep(1)
    print("\ntraining stopped")
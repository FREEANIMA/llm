from .config import Config

config = Config()


# UI가 그대로 렌더링할 수 있는 형태
MONITOR = {
    "stats": {
        "steps": 0,
        "loss": None,
        "gnorm": None,
        "lr": None,
        "sample_tags": [],          # list[str] — 배치 샘플 태그
        "mask_visible": [],         # list[int] — 각 query가 보는 key 수
        "key_padding": [],          # list[int] — 0/1
        "mask_preview": None,       # list[list[int]] — 작은 샘플 (예: 32x32)
    },
    "infer": {
        "tags": [],                 # list[str]
        "ids": [],                  # list[int] (원본)
        "updated_step": 0,
    },
    "status": {
        "training": False,
        "message": "",
    },
}
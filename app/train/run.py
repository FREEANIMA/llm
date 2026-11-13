import os
import json
import torch
import pandas as pd
from ..core.globals import config
from dataclasses import asdict
from ..model.model import Model
import threading
import queue
import torch.nn.functional as F
import random

class Train:
    def __init__(self, base_root):
        params = asdict(config)
        self.base_root = base_root

        self.params = params
        self.model = Model(**params).to(device=config.device, dtype=config.dtype)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            fused=torch.cuda.is_available(),
        )
        self.tags = self.mapping()
        self.stop_event = threading.Event()
        self.BATCH_Q = queue.Queue(maxsize=64)

    def mapping(self):
        with open(os.path.join(self.base_root, "data", "vocab.json"), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        metadata = pd.read_parquet(os.path.join(self.base_root, "data", "metadata.parquet"))
        tags_type = pd.read_parquet(os.path.join(self.base_root, "data", "tags_type.parquet"))

        BAD = {"", "NO_TAGS", "LOAD_FAIL", "NULL", "null"}

        df = (
            metadata[["tags"]]
            .dropna(subset=["tags"])
            .reset_index()
            .assign(tags=lambda d: d["tags"].str.split(", "))
            .explode("tags")
            .rename(columns={"tags": "tag"})
        )

        df["tag"] = df["tag"].str.strip()
        df = df[~df["tag"].isin(BAD)]
        df = df[df["tag"].notna()]

        df = df.merge(tags_type[["tag", "dup"]], on="tag", how="left")
        df["tag"] = df["dup"].fillna(df["tag"])
        df = df.drop(columns=["dup"])
        df = df.drop_duplicates(subset=["index", "tag"], keep="first")

        self.tags = (
            df.groupby("index")["tag"]
            .agg(list)
            .reindex(range(len(metadata)))
            .apply(lambda x: x if isinstance(x, list) else [])
            .tolist()
        )

        self.PAD_ID = self.vocab["<pad>"]
        UNK_ID = self.vocab["<unk>"]
        MAX_LEN = 256
        unknown = set()

        def to_ids_padded(xs, max_len=MAX_LEN):
            ids = []
            for t in xs:
                i = self.vocab.get(t)
                if i is None:
                    unknown.add(t)
                    i = UNK_ID
                ids.append(i)
                if len(ids) >= max_len:
                    break
            if len(ids) < max_len:
                ids.extend([self.PAD_ID] * (max_len - len(ids)))
            return ids

        self.tags = [to_ids_padded(xs) for xs in self.tags]

        if unknown:
            print(f"[mapping] vocab 미등록 태그 {len(unknown)}개: {list(unknown)[:10]}")

        return self.tags

    def batch_loader(self):
        N = len(self.tags)
        bs = config.batch_size
        pad_id = self.PAD_ID
        indices = list(range(N))

        while not self.stop_event.is_set():
            random.shuffle(indices)

            for start in range(0, N, bs):
                if self.stop_event.is_set():
                    return

                chunk = []
                # 2) 실제 토큰만 셔플, 패딩은 뒤로 유지
                for i in indices[start:start + bs]:
                    xs = self.tags[i]                    
                    valid = [t for t in xs if t != pad_id]
                    pads  = [t for t in xs if t == pad_id]
                    random.shuffle(valid)
                    chunk.append(valid + pads)

                self.BATCH_Q.put(torch.tensor(chunk, dtype=torch.long))

    def train(self):
        self.model.train()

        while not self.stop_event.is_set():
            try:
                batch = self.BATCH_Q.get(timeout=0.5)
            except queue.Empty:
                continue

            batch = batch.to(config.device, non_blocking=True)

            inputs = batch[:, :-1]
            target = batch[:, 1:]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=config.dtype, enabled=config.use_amp):
                logits = self.model.forward(x=inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=self.PAD_ID,
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            
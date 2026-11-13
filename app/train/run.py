import os
import json
import random
import queue
import threading

import torch
import torch.nn.functional as F
import pandas as pd

from dataclasses import asdict
from ..core.globals import config, MONITOR
from ..model.model import Model

class Train:
    def __init__(self, base_root):
        self.base_root = base_root

        with open(os.path.join(self.base_root, "data", "vocab.json"), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        ids = sorted(self.vocab.values())
        assert ids[0] == 0 and ids[-1] == len(ids) - 1, "vocab ID가 0..N-1 연속이어야 함"
        self.id_to_tag = {v: k for k, v in self.vocab.items()}
        
        self.PAD_ID = self.vocab["<pad>"]
        self.CLS_ID = self.vocab["<cls>"]
        self.EOS_ID = self.vocab["<eos>"]
        self.UNK_ID = self.vocab["<unk>"]
        
        params = asdict(config)
        params["vocab_size"] = len(self.vocab)
        self.params = params
        self.model = Model(**params).to(device=config.device, dtype=config.dtype)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            fused=torch.cuda.is_available(),
        )

        self.model_path = os.path.join(self.base_root, "weights", "model.pt")
        self.steps = self.model_load(self.model_path)

        self.stop_event = threading.Event()
        self.BATCH_Q = queue.Queue(maxsize=64)

        self.tags = self.mapping()

    def mapping(self):
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

        MAX_LEN = 256
        unknown = set()

        def to_ids_padded(xs, max_len=MAX_LEN):
            body = []
            for t in xs:
                i = self.vocab.get(t)
                if i is None:
                    unknown.add(t)
                    i = self.UNK_ID
                body.append(i)
                if len(body) >= max_len - 2:
                    break
            seq = [self.CLS_ID] + body + [self.EOS_ID]
            if len(seq) < max_len:
                seq.extend([self.PAD_ID] * (max_len - len(seq)))
            return seq

        self.tags = [to_ids_padded(xs) for xs in self.tags]

        if unknown:
            print(f"[mapping] vocab 미등록 태그 {len(unknown)}개: {list(unknown)[:10]}")

        return self.tags

    def batch_loader(self):
        N = len(self.tags)
        bs = config.batch_size
        pad_id = self.PAD_ID
        cls_id = self.CLS_ID
        eos_id = self.EOS_ID
        indices = list(range(N))

        while not self.stop_event.is_set():
            random.shuffle(indices)

            for start in range(0, N, bs):
                if self.stop_event.is_set():
                    return

                chunk = []
                for i in indices[start:start + bs]:
                    xs = self.tags[i]
                    eos_pos = len(xs) - 1
                    while eos_pos > 0 and xs[eos_pos] == pad_id:
                        eos_pos -= 1

                    tags_only = xs[1:eos_pos]
                    random.shuffle(tags_only)

                    n_pad = len(xs) - 2 - len(tags_only)
                    seq = [cls_id] + tags_only + [eos_id] + [pad_id] * n_pad
                    chunk.append(seq)

                batch = torch.tensor(chunk, dtype=torch.long)
                self.BATCH_Q.put(batch)

    def model_save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        checkpoint = {
            "steps": self.steps,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(config),
        }
        torch.save(checkpoint, self.model_path)

    def model_load(self, path):
        if not os.path.exists(path):
            print(f"[model_load] {path} 없음 → 새로 시작")
            return 0

        checkpoint = torch.load(path, map_location=config.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        saved_cfg = checkpoint.get("config")
        cur_cfg = asdict(config)
        if saved_cfg and saved_cfg != cur_cfg:
            print(f"[model_load] ⚠ config 불일치")
            print(f"  saved: {saved_cfg}")
            print(f"  cur  : {cur_cfg}")

        steps = checkpoint["steps"]
        print(f"[model_load] steps={steps} 로드 완료")
        return steps

    @torch.no_grad()
    def inference(self, max_new_tokens=256, temperature=1.0, top_k=None, use_amp=None):
        self.model.eval()
        device = config.device
        use_amp = config.use_amp if use_amp is None else use_amp

        # 시작 토큰: <cls>
        input_ids = torch.tensor([[self.CLS_ID]], dtype=torch.long, device=device)

        past_kv = None
        generated = []

        for _ in range(max_new_tokens):
            L_q = input_ids.shape[1]
            offset = 0 if past_kv is None else past_kv[0][0].shape[2]
            L_k = L_q + offset
            attn_mask = torch.ones(L_q, L_k, dtype=torch.bool, device=device).tril(diagonal=offset)
            attn_mask = attn_mask[None, None]   # (1,1,L_q,L_k)

            with torch.amp.autocast(device_type="cuda", dtype=config.dtype, enabled=use_amp):
                logits, past_kv = self.model.forward(
                    x=input_ids,
                    attn_mask=attn_mask,
                    past_kv=past_kv,
                    use_cache=True,
                )
            next_logits = logits[:, -1, :].float()

            if temperature and temperature > 0:
                next_logits = next_logits / temperature
            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                topk_vals, _ = torch.topk(next_logits, k)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() == self.EOS_ID:
                break

            generated.append(next_id.item())
            input_ids = next_id

        self.model.train()
        return generated

    def train(self):
        self.model.train()

        L_in_cached = None
        causal = None

        try:
            while not self.stop_event.is_set():
                try:
                    batch = self.BATCH_Q.get(timeout=0.5)
                except queue.Empty:
                    continue

                batch = batch.to(config.device, non_blocking=True)

                inputs = batch[:, :-1]
                target = batch[:, 1:]

                L_in = inputs.shape[1]
                if causal is None or L_in_cached != L_in:
                    causal = torch.ones(L_in, L_in, dtype=torch.bool, device=inputs.device).tril()
                    L_in_cached = L_in

                key_padding = (inputs != self.PAD_ID)
                attn_mask = causal[None, None] & key_padding[:, None, None, :]

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", dtype=config.dtype, enabled=config.use_amp):
                    logits = self.model.forward(x=inputs, attn_mask=attn_mask)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target.reshape(-1),
                        ignore_index=self.PAD_ID,
                    )

                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.steps += 1

                if self.steps % 10 == 0:
                    with torch.no_grad():
                        sample_mask = attn_mask[:1, 0].detach().cpu() 
                        visible = sample_mask.sum(dim=-1)
                        sample_pad = key_padding[:1].detach().cpu() 

                    MONITOR["stats"] = {
                        "loss": loss.item(),
                        "gnorm": gnorm.item(),
                        "steps": self.steps,
                        "tags_id": batch[:2].detach().cpu(),
                        "mask": sample_mask,
                        "mask_visible": visible,
                        "key_padding": sample_pad,
                    }

                if self.steps % 1000 == 0:
                    self.model_save()
                    generates =self.inference()

                    MONITOR["infer"] = [self.id_to_tag.get(i, f"<{i}>") for i in generates]

        except Exception:
            import traceback
            traceback.print_exc()
            self.stop_event.set()
            raise
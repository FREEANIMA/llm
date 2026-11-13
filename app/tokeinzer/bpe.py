import os
import json
import pandas as pd

class MakeTokenizer:
    def __init__(self, base_root):
        tags_type = os.path.join(base_root, "data", "tags_type.parquet")
        self.df = pd.read_parquet(tags_type)
        self.dup_df = self.df.loc[
            self.df["dup"].notna() & (self.df["dup"] != ""), ["tag", "dup"]
        ].reset_index(drop=True)

        self.eos_id = 1
        self.cls_id = 2
        self.pad_id = 3
        self.unk_id = 4

        self.vocab_path = os.path.join(base_root, "data", "vocab.json")

    def _save(self):
        specials = {
            "<eos>": self.eos_id,
            "<cls>": self.cls_id,
            "<pad>": self.pad_id,
            "<unk>": self.unk_id,
        }
        tags = self.dup_df["dup"].drop_duplicates().tolist()

        vocab = dict(specials)
        start = max(specials.values()) + 1
        for i, tag in enumerate(tags, start=start):
            vocab[tag] = i

        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=1)
        return len(vocab)

make = MakeTokenizer(base_root=r"C:\Users\HONG\Desktop\inquiry\llm\app")._save()
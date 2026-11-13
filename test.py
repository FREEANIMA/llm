import os
import json
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm          # 노트북/터미널 자동 선택

def mapping(base_root, verbose=True, drop_empty=True):
    with open(os.path.join(base_root, "data", "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    metadata = pd.read_parquet(os.path.join(base_root, "data", "metadata.parquet"))
    tags_type = pd.read_parquet(os.path.join(base_root, "data", "tags_type.parquet"))

    BAD = {"NO_TAGS", "LOAD_FAIL"}

    # --- 사전 검증 ---
    if verbose:
        tqdm.write(f"[mapping] metadata rows: {len(metadata)}")
        tqdm.write(f"[mapping] tags_type rows: {len(tags_type)}")

    dup_rows = tags_type["tag"].duplicated().sum()
    if dup_rows:
        if verbose:
            tqdm.write(f"[mapping] ⚠ tags_type['tag'] 중복 {dup_rows}개 → 첫 값만 사용")
        tags_type = tags_type.drop_duplicates("tag", keep="first")

    if verbose and tags_type["dup"].isna().any():
        tqdm.write(f"[mapping] ⚠ tags_type['dup'] NaN {tags_type['dup'].isna().sum()}개 → 원본 유지")

    # --- explode ---
    with tqdm(total=1, desc="[mapping] explode", disable=not verbose) as pbar:
        df = (
            metadata[["tags"]]
            .dropna(subset=["tags"])
            .reset_index()
            .assign(tags=lambda d: d["tags"].str.split(", "))
            .explode("tags")
            .rename(columns={"tags": "tag"})
        )
        pbar.update(1)

    # --- BAD 제거 ---
    df = df[~df["tag"].isin(BAD)]

    # --- dup 매핑 ---
    with tqdm(total=1, desc="[mapping] merge dup", disable=not verbose) as pbar:
        df = df.merge(tags_type[["tag", "dup"]], on="tag", how="left")
        df["tag"] = df["dup"].fillna(df["tag"])
        df = df.drop(columns=["dup"])
        pbar.update(1)

    # --- 미매핑 태그 통계 ---
    if verbose:
        all_tags = set(metadata["tags"].dropna().str.split(", ").explode())
        unknown = all_tags - set(tags_type["tag"]) - BAD
        tqdm.write(f"[mapping] tags_type에 없는 태그: {len(unknown)}개")
        if unknown:
            tqdm.write(f"  예시: {list(unknown)[:10]}")

    # --- (원본 행, 대표 태그) 기준 중복 제거 ---
    df = df.drop_duplicates(subset=["index", "tag"], keep="first")

    # --- 원본 행별 리스트 복원 ---
    with tqdm(total=1, desc="[mapping] groupby", disable=not verbose) as pbar:
        tags = (
            df.groupby("index")["tag"]
            .agg(list)
            .reindex(range(len(metadata)))
            .apply(lambda x: x if isinstance(x, list) else [])
            .tolist()
        )
        pbar.update(1)

    if drop_empty:
        tags = [t for t in tags if t]

    # --- 통계 ---
    if verbose:
        lens = [len(t) for t in tags]
        if lens:
            tqdm.write(f"[mapping] 결과 리스트 수: {len(tags)}")
            tqdm.write(f"[mapping] 평균 태그 수: {sum(lens)/len(lens):.2f}")
            tqdm.write(f"[mapping] 최대 태그 수: {max(lens)}")
            tqdm.write(f"[mapping] 빈 리스트: {sum(1 for t in tags if not t)}개")
            top = Counter(t for ts in tags for t in ts).most_common(10)
            tqdm.write(f"[mapping] 상위 태그: {top}")

    return tags


if __name__ == "__main__":
    base_root = r"C:\Users\HONG\Desktop\inquiry\llm\app"
    tags = mapping(base_root)
    print(tags[:5])
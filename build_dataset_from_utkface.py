# -*- coding: utf-8 -*-
"""
建立数据集：从 UTKFace 人脸库生成 data/train、data/val、data/test

数据来源（本机可直连的镜像）：
    https://hf-mirror.com/datasets/nlphuji/utk_faces
    - utk_faces_images.zip      23708 张 200x200 彩色人脸
    - utk_dataset_metadata.csv  每张图的 age / gender / race 标注（gender: male/female）

本脚本做的事：
    1. 只保留 16~70 岁的成年人样本；
    2. 按"种族"分层抽样，让男/女两类的种族构成一致
       —— 防止 SVM 走捷径学到"肤色→性别"这种假规律；
    3. 切成 train / val / test 三份（互不重叠），按固定命名习惯存成
       data/train/male/male_0000.jpg 这种形式；
    4. 同时输出 data/dataset_manifest.csv，记录每张图的年龄/种族/来源文件名（便于核对数据来源）。

运行： python build_dataset_from_utkface.py            # 已有数据则跳过
      python build_dataset_from_utkface.py --force    # 删掉重建
      python build_dataset_from_utkface.py --train-per-class 400 --val-per-class 60 --test-per-class 100

不需要先把 zip 解压，脚本直接从 zip 里读图，省 130MB 磁盘。
"""

import os
import csv
import zlib
import zipfile
import argparse
from collections import Counter

import cv2
import numpy as np
import pandas as pd

# ===================== 配置 =====================
RAW_DIR = '_dataset_raw'
ZIP_NAME = 'utk_faces_images.zip'
CSV_NAME = 'utk_dataset_metadata.csv'
ZIP_PREFIX = 'utk_faces_images/'          # zip 内部的目录名

DATA_DIR = 'data'
SPLITS = ['train', 'val', 'test']
CLASS_NAMES = ['male', 'female']

AGE_MIN, AGE_MAX = 16, 70                  # 只取成年人（避免小朋友/百岁老人干扰）
RACES = ['white', 'black', 'indian', 'asian', 'others_hispanic_latino_middle_eastern']

# 每类（男/女）各划分多少张
DEFAULT_COUNTS = {'train': 800, 'val': 200, 'test': 200}
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description='从 UTKFace 构建性别识别数据集')
    p.add_argument('--force', action='store_true', help='目标目录已存在时也重建')
    p.add_argument('--train-per-class', type=int, default=DEFAULT_COUNTS['train'])
    p.add_argument('--val-per-class', type=int, default=DEFAULT_COUNTS['val'])
    p.add_argument('--test-per-class', type=int, default=DEFAULT_COUNTS['test'])
    return p.parse_args()


def get_zip_path():
    """找到 zip（默认在 _dataset_raw 下，也允许放在项目根目录）"""
    for path in (os.path.join(RAW_DIR, ZIP_NAME), ZIP_NAME):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"没找到 {ZIP_NAME}！请先下载：\n"
        f"  curl -L -o {RAW_DIR}/{ZIP_NAME} "
        f"https://hf-mirror.com/datasets/nlphuji/utk_faces/resolve/main/{ZIP_NAME}\n"
        f"  curl -L -o {RAW_DIR}/{CSV_NAME} "
        f"https://hf-mirror.com/datasets/nlphuji/utk_faces/resolve/main/{CSV_NAME}"
    )


def get_csv_path():
    for path in (os.path.join(RAW_DIR, CSV_NAME), CSV_NAME):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"没找到 {CSV_NAME}，下载命令见 build_dataset_from_utkface.py 顶部注释")


def sample_balanced_by_race(pool, n, seed):
    """
    从 pool 里抽 n 条，尽量让抽出来的样本「种族构成」和整体一致。
    某个种族不够时，把缺额再分给还有余量的种族。
    """
    if n >= len(pool):
        return pool

    rng = np.random.RandomState(seed)
    props = pool['race'].value_counts(normalize=True)
    quota = {r: int(round(n * props.get(r, 0.0))) for r in RACES}
    # 按比例取完后剩下的零头，从最多的种族里补
    quota[max(quota, key=quota.get)] += n - sum(quota.values())

    picked, shortage = [], 0
    for r in RACES:
        sub = pool[pool['race'] == r]
        want = min(quota.get(r, 0), len(sub))
        shortage += quota.get(r, 0) - want
        if want > 0:
            picked.append(sub.sample(n=want, random_state=rng.randint(1 << 30)))

    # 把没取够的名额补给还有余量的种族
    if shortage > 0:
        used = set().union(*[set(p.index) for p in picked]) if picked else set()
        spare = pool[~pool.index.isin(used)].sample(
            n=min(shortage, len(pool) - len(used)), random_state=rng.randint(1 << 30)
        )
        picked.append(spare)

    return pd.concat(picked).sample(frac=1, random_state=seed).head(n)


def main():
    args = parse_args()
    counts = {'train': args.train_per_class, 'val': args.val_per_class,
              'test': args.test_per_class}

    zip_path, csv_path = get_zip_path(), get_csv_path()
    print(f"读取标注: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[(df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX)]
    df = df[df['gender'].isin(CLASS_NAMES)]
    print(f"16~{AGE_MAX} 岁样本: {len(df)} 张 "
          f"(男 {int((df.gender == 'male').sum())} / 女 {int((df.gender == 'female').sum())})")

    # 目标目录
    for split in SPLITS:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)
    already = sum(len(os.listdir(os.path.join(DATA_DIR, s, c)))
                  for s in SPLITS for c in CLASS_NAMES)
    if already > 0:
        if not args.force:
            print(f"\ndata/ 里已经有 {already} 张图片了，直接跳过。")
            print("要重新生成请加 --force（会先清空 data/{train,val,test} 下的旧图片）")
            return
        for split in SPLITS:
            for cls in CLASS_NAMES:
                for f in os.listdir(os.path.join(DATA_DIR, split, cls)):
                    os.remove(os.path.join(DATA_DIR, split, cls, f))
        print(f"已清空旧数据（{already} 张），开始重建...")

    zf = zipfile.ZipFile(zip_path)
    available = set(zf.namelist())
    manifest = []
    total_written = total_failed = 0

    for cls in CLASS_NAMES:
        pool = df[df['gender'] == cls]
        print(f"\n===== {cls}（可选 {len(pool)} 张）=====")

        for split in SPLITS:
            n = counts[split]
            if n <= 0:
                continue
            # 三个 split 依次从 pool 里「抽完即删」，保证互不重叠
            # （用 crc32 而不是内置 hash，保证每次运行抽样结果完全一样）
            seed = zlib.crc32(f'{cls}-{split}-{SEED}'.encode()) % (1 << 30)
            picked = sample_balanced_by_race(pool, n, seed=seed)
            pool = pool.drop(index=picked.index)

            out_dir = os.path.join(DATA_DIR, split, cls)
            wrote = 0
            wrote_races = Counter()
            for _, row in picked.iterrows():
                member = f"{ZIP_PREFIX}{row['image_name']}.chip.jpg"
                if member not in available:
                    total_failed += 1
                    continue
                try:
                    img = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    img = None
                if img is None:
                    total_failed += 1
                    continue

                out_name = f"{cls}_{wrote:04d}.jpg"
                cv2.imwrite(os.path.join(out_dir, out_name), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest.append({
                    'split': split, 'class': cls, 'file': os.path.join(split, cls, out_name),
                    'age': int(row['age']), 'race': row['race'],
                    'source_image': row['image_name'],
                })
                wrote += 1
                wrote_races[row['race']] += 1

            total_written += wrote
            # 打印这一份的种族构成，方便确认没偏
            print(f"  {split:5s}: 写入 {wrote:4d} 张  种族构成 {dict(wrote_races)}")

    zf.close()

    # 保存清单
    manifest_path = os.path.join(DATA_DIR, 'dataset_manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'class', 'file', 'age', 'race', 'source_image'])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\n数据集构建完成：共 {total_written} 张图片"
          f"{f'（{total_failed} 张读取失败已跳过）' if total_failed else ''}")
    print(f"清单已保存: {manifest_path}")
    print(f"目录结构: {DATA_DIR}/{{{'|'.join(SPLITS)}}}/{{{'|'.join(CLASS_NAMES)}}}/")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
生成仅包含干净文件的发布压缩包（白名单方式，不会打包数据集、虚拟环境和生成物）。

用法：
    python package_clean.py                 # 源码 + 模型 + 文档

说明：
    压缩包内会附带 MANIFEST.txt，列出每个文件的大小与 SHA-256。
"""

import hashlib
import sys
import zipfile
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / 'dist'

# 白名单：只打包这些文件
BASE_FILES = [
    'README.md',
    'LICENSE',
    'NOTICE',
    'requirements.txt',
    '.gitignore',
    'gender_classifier.py',
    'realtime_gender_detection.py',
    'collect_gender_data.py',
    'build_dataset_from_utkface.py',
    'CNN_gender.py',
    'gender_classifier_model.pkl',
    'model_info.pkl',
    'package_clean.py',
]

# 仅用于提示，白名单机制本身已经保证不会打包这些内容
EXCLUDED_HINT = [
    '.venv/（含本机用户名与绝对路径）',
    'data/（2400 张人脸）',
    '_dataset_raw/（原始人脸压缩包）',
    'results/（含人脸的生成图）',
    '__pycache__/、*.pyc',
    'dist/、gender_shot_*.jpg',
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = list(BASE_FILES)

    missing = [f for f in files if not (ROOT / f).is_file()]
    if missing:
        print('错误：以下文件不存在，无法打包：')
        for f in missing:
            print('  -', f)
        return 1

    DIST_DIR.mkdir(exist_ok=True)
    zip_path = DIST_DIR / 'lesson04_clean.zip'

    manifest = [
        '项目：基于机器视觉的人体性别识别',
        '打包时间：' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '',
        '文件清单（SHA-256  字节数  路径）：',
    ]

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            path = ROOT / rel
            data = path.read_bytes()
            # 固定时间戳，避免压缩包泄漏文件修改时间
            info = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
            manifest.append(f'{hashlib.sha256(data).hexdigest()}  {len(data):>10}  {rel}')

        manifest += ['', '未打包内容（按设计排除）：'] + [f'- {x}' for x in EXCLUDED_HINT]
        zf.writestr('MANIFEST.txt', '\n'.join(manifest) + '\n')

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f'已生成：{zip_path}（{size_mb:.2f} MB，{len(files) + 1} 个条目）')
    for rel in files:
        print('  +', rel)
    print('  + MANIFEST.txt')
    print('\n未打包：' + '；'.join(EXCLUDED_HINT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

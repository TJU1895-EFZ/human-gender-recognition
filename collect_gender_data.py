# -*- coding: utf-8 -*-
"""
第一部分：采集数据（用摄像头拍自己的照片，按性别分到 data 文件夹）

摄像头采集逻辑：
    - 按 m 开始录"男"，按 f 开始录"女"，选中后自动连续拍照（每 0.5 秒一张）；
    - 每 5 张里 4 张进 data/train、1 张进 data/test（8:2 划分）；
    - 画面上实时显示状态、已拍张数、以及 Haar 检测到的人脸框；
    - 没检测到人脸时不保存（防止把天花板、桌面这些没脸的画面当成训练数据）。

运行： python collect_gender_data.py
      python collect_gender_data.py --every 0.3          # 每 0.3 秒拍一张
      python collect_gender_data.py --no-require-face    # 不管有没有人脸都拍
按键： m=录男性  f=录女性  s=停止/暂停  q=退出

隐私提示：
    采集前请确保已获得被拍摄者同意。人脸属于敏感生物识别信息，
    采集到的 data/ 目录已被 .gitignore 排除，请勿上传或用于未经授权的用途。
"""

import os
import sys
import time
import argparse

import cv2

# OpenCV 5.0 起删掉了 Haar 级联，提前给出能看懂的报错（详见 gender_classifier.py）
if not hasattr(cv2, 'CascadeClassifier'):
    raise ImportError(
        f"当前 opencv-python 版本是 {cv2.__version__}，OpenCV 5.0 起移除了 Haar 级联，\n"
        f"人脸检测跑不了。请在同一个环境里装回 4.x：\n"
        f'    "{sys.executable}" -m pip install "opencv-python==4.11.0.86"\n'
    )

CLASS_NAMES = ['male', 'female']
CLASS_CN = {'male': '男', 'female': '女'}
KEY_MAP = {ord('m'): 'male', ord('f'): 'female'}
TRAIN_RATIO_MOD = 5          # 每 5 张里 1 张进测试集

_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description='摄像头采集性别训练数据')
    p.add_argument('--data-dir', default='data', help='数据集根目录（默认 data）')
    p.add_argument('--camera', type=int, default=0, help='摄像头编号（默认 0）')
    p.add_argument('--every', type=float, default=0.5, help='自动拍照间隔秒数（默认 0.5）')
    p.add_argument('--no-require-face', action='store_true',
                   help='即使没检测到人脸也保存（默认不保存）')
    return p.parse_args()


def next_index(class_dir):
    """接着已有文件继续编号，避免覆盖之前拍的照片"""
    if not os.path.exists(class_dir):
        return 0
    idx = [int(f.split('_')[-1].split('.')[0])
           for f in os.listdir(class_dir)
           if f.lower().endswith('.jpg') and f.split('_')[-1].split('.')[0].isdigit()]
    return max(idx) + 1 if idx else 0


def detect_faces(frame):
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))


def main():
    args = parse_args()
    base_dir = args.data_dir

    # 建好目录：data/train/male、data/train/female、data/test/male、data/test/female
    for split in ('train', 'test'):
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)   # Windows 上加 DSHOW 打开更快
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查摄像头是否被占用或驱动是否正常")
        return

    print("摄像头已打开")
    print(f"按 m 录【男】  f 录【女】  s 暂停  q 退出")
    print(f"自动拍照间隔 {args.every} 秒，每 {TRAIN_RATIO_MOD} 张里 1 张放进测试集")

    current_class = None
    # 编号从 train/test 两个目录里的最大值往后接，避免覆盖以前拍的照片
    counters = {cls: max(next_index(os.path.join(base_dir, 'train', cls)),
                         next_index(os.path.join(base_dir, 'test', cls)))
                for cls in CLASS_NAMES}
    split_counter = 0                 # 用于 train/test 的 8:2 轮换
    saved_total = 0
    last_shot = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头画面")
            break

        faces = detect_faces(frame)
        now = time.time()

        # 是否要保存这一帧
        if current_class and (now - last_shot) >= args.every:
            if faces or args.no_require_face:
                split = 'test' if split_counter % TRAIN_RATIO_MOD == 0 else 'train'
                idx = counters[current_class]
                photo_name = f'{current_class}_{idx:04d}.jpg'
                save_path = os.path.join(base_dir, split, current_class, photo_name)
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                counters[current_class] += 1
                split_counter += 1
                saved_total += 1
                last_shot = now
                print(f"保存: {save_path}")
            else:
                last_shot = now      # 没脸就跳过这一轮，别刷屏

        # 画界面
        if current_class:
            status = f"Recording {current_class}({CLASS_CN[current_class]})"
            color = (0, 255, 0)
        else:
            status = "Press m=male / f=female"
            color = (0, 255, 255)
        cv2.putText(frame, status, (10, 30), _FONT, 0.7, color, 2)
        cv2.putText(frame, f"Saved: {saved_total}", (10, 60), _FONT, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"male:{counters['male']}  female:{counters['female']}",
                    (10, 90), _FONT, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "m=male f=female s=stop q=quit", (10, 120), _FONT, 0.6,
                    (200, 200, 200), 1)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if current_class and not faces and not args.no_require_face:
            cv2.putText(frame, "No face detected - not saving", (10, 150), _FONT, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow('Gender Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in KEY_MAP:
            current_class = KEY_MAP[key]
            print(f"开始录制 {current_class}（{CLASS_CN[current_class]}）类别...")
        elif key == ord('s'):
            current_class = None
            print("已暂停")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n数据收集完成! 本次共保存 {saved_total} 张")
    for cls in CLASS_NAMES:
        n_train = len([f for f in os.listdir(os.path.join(base_dir, 'train', cls))
                       if f.lower().endswith('.jpg')])
        n_test = len([f for f in os.listdir(os.path.join(base_dir, 'test', cls))
                      if f.lower().endswith('.jpg')])
        print(f"  {cls}({CLASS_CN[cls]}): train {n_train} 张, test {n_test} 张")
    print("\n下一步：python gender_classifier.py   （训练模型）")


if __name__ == "__main__":
    main()

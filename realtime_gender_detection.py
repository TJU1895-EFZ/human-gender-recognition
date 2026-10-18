# -*- coding: utf-8 -*-
"""
第三部分：验证模型（调用第二部分训练好的 gender_classifier_model.pkl，实时识别性别）

功能：
    - 每帧画面送进分类器，画面上显示类别 + 置信度 + 各类概率，按 q 退出；
    - 区别是这里先用 Haar 把人脸框出来再判断（性别在脸上，整帧直接喂给模型会被背景干扰），
      并且用 PIL 在画面上写中文（cv2.putText 不支持中文）。

运行：
    python realtime_gender_detection.py                      # 打开摄像头实时识别
    python realtime_gender_detection.py --image 某张照片.jpg   # 只测一张图片
    python realtime_gender_detection.py --dir data/test/female # 批量测一个文件夹
    python realtime_gender_detection.py --image x.jpg --no-show  # 不弹窗，只打印结果
"""

import os
import sys
import time
import argparse
import glob

import cv2
import numpy as np

# 下面要 print 的 ✅/❌/💡 在 GBK 控制台（cmd、PyCharm 运行窗口、输出被重定向时）编不出来，
# 会直接 UnicodeEncodeError 把程序打断。这里允许替换成 "?"，不要为了一个表情崩掉。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

# cv2.imshow 的窗口标题在 Windows 上写中文会变成乱码，所以先建英文标题的窗口，
# 再用 win32gui 把标题改成中文。没装 pywin32 就保持英文标题，不影响识别功能。
try:
    import win32gui
except ImportError:
    win32gui = None


def set_window_title(old_title, new_title):
    """把 cv2 窗口的标题从 old_title 改成 new_title（中文），失败时静默跳过"""
    if win32gui is None:
        return
    handle = win32gui.FindWindow(None, old_title)
    if handle:
        win32gui.SetWindowText(handle, new_title)


# 把当前目录加进搜索路径，保证能 import 到 gender_classifier
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from gender_classifier import GenderClassifier, CLASS_NAMES_CN, detect_face_box

# ===================== 中文写图（cv2.putText 画不了中文，借 PIL 一下） =====================
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r'C:\Windows\Fonts\msyh.ttc',       # 微软雅黑
    r'C:\Windows\Fonts\simhei.ttf',     # 黑体
    '/System/Library/Fonts/PingFang.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
]
_FONT_CACHE = {}


def get_font(size=26):
    if size not in _FONT_CACHE:
        font = None
        for path in FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    pass
        _FONT_CACHE[size] = font or ImageFont.load_default()
    return _FONT_CACHE[size]


def imread_unicode(path):
    """
    读图片，兼容中文路径。
    cv2.imread 在 Windows 上只吃 ANSI 路径，遇到中文/日文路径直接返回 None
    （项目放在中文目录下用绝对路径就会踩到）；
    退回 np.fromfile + cv2.imdecode 就能正常读。
    """
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img_bgr):
    """写图片，同样是 cv2.imwrite 不支持中文路径时的兜底写法"""
    try:
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or '.jpg', img_bgr)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img_bgr)


def draw_text_cn(img_bgr, text, org, color_bgr=(0, 255, 0), size=26):
    """在 BGR 图上写中文文字（原地修改并返回）"""
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
    draw.text(org, text, font=get_font(size), fill=(r, g, b))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ===================== 画检测结果 =====================
def annotate(frame, result, show_probs=True):
    """把预测结果画到画面上：人脸框 + 类别 + 置信度 + 各类概率"""
    if not result:
        return frame

    label = f"{result['class_cn']} {result['class']}  {result['confidence'] * 100:.1f}%"
    color = (0, 200, 0) if result['class'] == 'male' else (200, 80, 0)   # 男=绿 女=蓝

    box = result.get('face_box')
    if not box:
        # 没框到脸就别硬报性别了（这时模型是拿整帧瞎猜的），提示把脸对准镜头
        return draw_text_cn(frame, "未检测到人脸，请把脸对准镜头", (10, 30), (0, 0, 255), 24)

    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    ty = max(0, y - 34)
    frame = draw_text_cn(frame, label, (x + 4, ty), color, 28)

    if show_probs:
        y_off = ty + 34
        for cls_name, prob in result['all_probabilities'].items():
            text = f"{CLASS_NAMES_CN.get(cls_name, cls_name)}({cls_name}): {prob:.3f}"
            frame = draw_text_cn(frame, text, (x + 4, y_off), (255, 255, 255), 20)
            y_off += 24
    return frame


# ===================== 三种运行模式 =====================
WIN_NAME = "temp"        # cv2 窗口的英文标题，建好后再用 win32gui 改成中文
WIN_TITLE_CN = "开始实时识别性别"


def run_camera(classifier, camera=0, show=True, max_frames=0):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("错误：无法打开摄像头（可以先用 --image 测试一张图片）")
        return

    if show:
        # 窗口只建一次、标题也只改一次：改完标题后 FindWindow("temp") 就找不到它了，
        # 而且窗口每帧销毁重建的话画面会一直闪
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WIN_NAME, np.zeros((120, 320, 3), dtype=np.uint8))
        cv2.waitKey(1)
        set_window_title(WIN_NAME, WIN_TITLE_CN)

    print(f"开始实时性别识别，按 q 退出{'，按 s 截图' if show else ''}")
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取图像")
            break

        t0 = time.time()
        result = classifier.predict_image(frame)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        frame = annotate(frame, result)
        frame = draw_text_cn(frame, f"FPS {fps:.1f}", (10, frame.shape[0] - 30),
                             (255, 255, 0), 20)
        n += 1

        if show:
            cv2.imshow(WIN_NAME, frame)
            # waitKey 是唯一会处理窗口消息的地方，必须每帧调一次；
            # 参数给 0 会一直等到按键才返回，实时预览就变成"按一下走一帧"了
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):       # q 或 ESC 退出
                break
            elif key == ord('s'):
                shot = f"gender_shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                if imwrite_unicode(shot, frame):
                    print(f"截图已保存: {shot}")
                else:
                    print(f"截图保存失败: {shot}")
            # 点到窗口右上角的叉也要能退出
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        elif n % 30 == 0:
            print(f"[{n}] {result['class_cn'] if result else '?'} "
                  f"{result['confidence'] * 100 if result else 0:.1f}%")

        if max_frames and n >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(classifier, image_path, show=True):
    img = imread_unicode(image_path)
    if img is None:
        print(f"图片加载失败，请检查路径: {image_path}")
        return None
    result = classifier.predict_image(img)
    if result is None:
        return None
    print(f"图片: {image_path}")
    print(f"预测性别: {result['class']}（{result['class_cn']}）  置信度: {result['confidence']:.3f}")
    for cls_name, prob in result['all_probabilities'].items():
        print(f"  {cls_name}: {prob:.3f}")
    if show:
        # 同样先建英文标题的窗口再改成中文（cv2 直接传中文标题在 Windows 上是乱码）
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WIN_NAME, annotate(img, result))
        set_window_title(WIN_NAME, '预测结果')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result


def run_dir(classifier, dir_path, show=False):
    """批量跑一个文件夹，直接给出准确率（文件夹名当真实标签）"""
    files = sorted(f for f in glob.glob(os.path.join(dir_path, '*'))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    if not files:
        print(f"文件夹里没有图片: {dir_path}")
        return
    true_class = os.path.basename(os.path.normpath(dir_path)).lower()   # male / female
    correct = 0
    print(f"批量测试 {len(files)} 张图片，真实类别 = {true_class}")
    for path in files:
        img = imread_unicode(path)
        if img is None:
            continue
        result = classifier.predict_image(img)
        if result is None:
            continue
        ok = result['class'] == true_class
        correct += ok
        mark = '√' if ok else '×'
        print(f"  {mark} {os.path.basename(path)}  预测 {result['class']}"
              f" ({result['confidence']:.3f})")
    print(f"准确率: {correct}/{len(files)} = {correct / len(files):.3f}")


def main():
    parser = argparse.ArgumentParser(description='实时性别识别')
    parser.add_argument('--image', help='测试单张图片')
    parser.add_argument('--dir', help='批量测试一个文件夹（文件夹名当标签）')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-show', action='store_true', help='不弹窗，只打印结果')
    parser.add_argument('--frames', type=int, default=0, help='摄像头模式最多跑多少帧（0=不限）')
    args = parser.parse_args()

    classifier = GenderClassifier()

    # 先试一张全黑图触发模型加载，判断 pkl 在不在
    print("尝试加载模型...")
    probe = np.zeros((200, 200, 3), dtype=np.uint8)
    if classifier.predict_image(probe) is None:
        print("❌ 模型加载失败，请先运行 gender_classifier.py 训练模型")
        print("💡 确保以下文件存在:")
        print("   - gender_classifier_model.pkl")
        print("   - model_info.pkl")
        return
    info = classifier.model_info
    print(f"✅ 模型加载成功! 训练时间 {info.get('trained_at', '未知')}，"
          f"训练集准确率 {info.get('train_accuracy', 0):.3f}，"
          f"测试集准确率 {info.get('test_accuracy') or 0:.3f}")

    show = not args.no_show
    if args.image:
        run_image(classifier, args.image, show)
    elif args.dir:
        run_dir(classifier, args.dir, show)
    else:
        run_camera(classifier, args.camera, show, args.frames)


if __name__ == "__main__":
    main()

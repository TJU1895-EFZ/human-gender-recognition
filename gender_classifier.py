# -*- coding: utf-8 -*-
"""
第二部分：训练性别识别模型

流程：提取人脸特征 -> 训练 -> 保存 pkl -> 单张预测：

    1. 先用 Haar 级联检测人脸，把脸抠出来（性别信息在脸上，不在背景里）；
    2. 特征 = HOG（形状/纹理） + 32x32 灰度缩略图（五官结构） + HSV 颜色直方图（肤色/发色信息）；
    3. 用 sklearn 的 SVM(RBF) 训练，整条流水线（标准化 + PCA + SVM）一起存成 pkl，
       验证时直接加载，不需要 torch/tensorflow。

运行： python gender_classifier.py
输出： gender_classifier_model.pkl、model_info.pkl、results/*.png
"""

import os
import sys
import glob
import time
import pickle
import tempfile

import cv2
import numpy as np
import joblib

# OpenCV 5.0 起删掉了 Haar 级联（连 cv2.CascadeClassifier 都没有了），
# 直接报 AttributeError 太难看懂，这里提前拦住说清楚怎么修
if not hasattr(cv2, 'CascadeClassifier'):
    raise ImportError(
        f"当前 opencv-python 版本是 {cv2.__version__}，OpenCV 5.0 起移除了人脸检测用的 Haar 级联\n"
        f"（cv2.CascadeClassifier 不存在），本项目的人脸检测跑不了。\n"
        f"请在同一个环境里装回 4.x：\n"
        f'    "{sys.executable}" -m pip install "opencv-python==4.11.0.86"\n'
    )

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===================== Matplotlib 中文显示 =====================
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',      # Windows 优先
    'PingFang SC',          # macOS 优先
    'WenQuanYi Micro Hei',  # Linux 优先
    'DejaVu Sans'           # 兜底字体
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# ===================== 1. 配置参数 =====================
IMG_SIZE = (64, 64)          # HOG 输入尺寸
PIX_SIZE = 32                # 灰度缩略图尺寸
HIST_BINS = 32               # 每个颜色通道的直方图 bin 数

# 类别顺序固定为 ['male', 'female']，和 data/train 下的文件夹名一致
CLASS_NAMES = ['male', 'female']
CLASS_NAMES_CN = {'male': '男', 'female': '女'}

TRAIN_DIR = os.path.join('data', 'train')
TEST_DIR = os.path.join('data', 'test')

MODEL_PATH = 'gender_classifier_model.pkl'
INFO_PATH = 'model_info.pkl'
RESULT_DIR = 'results'

# 人脸检测器（opencv-python 自带，无需另外下载）
def _ascii_temp_path(filename):
    """找一个路径全是 ASCII 的可写目录来放级联 xml（cv2 只认 ANSI 路径）"""
    for d in (tempfile.gettempdir(),
              os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'Temp')):
        path = os.path.join(d, filename)
        try:
            path.encode('ascii')
            os.makedirs(d, exist_ok=True)
            return path
        except (UnicodeEncodeError, OSError):
            continue
    return None


def _load_cascade(xml_name):
    """
    加载 Haar 级联 xml。

    cv2.CascadeClassifier 内部是按 ANSI 路径打开文件的，项目放在中文目录下时
    读不到文件，而且不报错——只是静默变成一个空分类器，
    之后 detectMultiScale 直接抛 `!empty() in function 'cv::CascadeClassifier::detectMultiScale'`。
    所以这里先判断路径是不是纯 ASCII，不是就把 xml 复制到一个英文临时目录再加载。
    """
    src = os.path.join(cv2.data.haarcascades, xml_name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"找不到人脸检测级联文件: {src}")

    path = src
    try:
        path.encode('ascii')
    except UnicodeEncodeError:
        path = _ascii_temp_path(xml_name)
        if path is None:
            raise RuntimeError(
                "项目路径含中文，且找不到纯英文的临时目录来放人脸检测模型。\n"
                "请把项目移到纯英文路径下，或设置 TEMP 环境变量指向英文目录。"
            )
        with open(src, 'rb') as fin, open(path, 'wb') as fout:
            fout.write(fin.read())

    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"人脸检测级联加载失败: {path}")
    return cascade


_CASCADE = _load_cascade('haarcascade_frontalface_default.xml')

# HOG 描述子：窗口 64x64，块 16x16，步长 8x8，胞元 8x8，9 个方向 bin
_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
HOG_DIM = _HOG.getDescriptorSize()          # 1764
FEATURE_DIM = HOG_DIM + PIX_SIZE * PIX_SIZE + 3 * HIST_BINS   # 2884


# ===================== 2. 人脸检测 + 特征提取 =====================
def detect_face_box(image):
    """用 Haar 级联检测人脸，返回最大的那个人脸框 (x, y, w, h)；没检测到返回 None"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # 直方图均衡，提升光照差时的检出率
    faces = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    # 取面积最大的框，避免把背景里的人脸/误检当主体
    return max(faces, key=lambda f: f[2] * f[3])


def crop_face(image, box=None, margin=0.2):
    """
    抠出人脸区域，并向外扩 margin 比例（把下巴/头发也带一点，训练和预测保持一致）。
    没检测到人脸时返回原图（UTKFace 本身已经是裁剪过的人脸，退回整图影响不大）。
    传了 box 就直接用它，不再重复检测人脸（实时识别时每帧能省一次检测）。
    """
    if box is None:
        box = detect_face_box(image)
    if box is None:
        return image, False
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(image.shape[1], x + w + mx), min(image.shape[0], y + h + my)
    return image[y0:y1, x0:x1], True


def extract_features(image, use_face_detect=True):
    """检测人脸 + 抠脸 + 提特征（加载训练集时用；推理走 predict_image）"""
    face, _ = crop_face(image) if use_face_detect else (image, False)
    return features_from_face(face)


def features_from_face(face):
    """对已经抠好的人脸图计算特征（不含人脸检测，训练和推理共用同一个函数，保证一致）"""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # 特征1：HOG（形状/纹理，眼睛鼻子嘴巴的边缘走向）
    hog = _HOG.compute(cv2.resize(gray, IMG_SIZE)).flatten()

    # 特征2：32x32 灰度缩略图（五官整体结构）
    pix = cv2.resize(gray, (PIX_SIZE, PIX_SIZE)).astype(np.float32).flatten() / 255.0

    # 特征3：HSV 颜色直方图（描述肤色与发色）
    hsv = cv2.cvtColor(cv2.resize(face, IMG_SIZE), cv2.COLOR_BGR2HSV)
    hists = []
    for i in range(3):  # H, S, V 三个通道
        hist = cv2.calcHist([hsv], [i], None, [HIST_BINS], [0, 256])
        hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
        hists.append(hist)
    hist_feat = np.concatenate(hists)

    return np.concatenate([hog, pix, hist_feat]).astype(np.float32)


# ===================== 3. 分类器 =====================
class GenderClassifier:
    """性别分类器：标准化 + PCA + RBF-SVM 流水线"""

    def __init__(self, n_components=300, C=3):
        self.class_names = CLASS_NAMES
        # 标准化 + PCA 降维 + RBF-SVM：整条流水线一起存进 pkl，加载后即可直接预测
        # PCA(300) 和 C=3 是在 2400 张数据上扫出来的（见 README「四、结果」），
        # 概率用 CalibratedClassifierCV 校准（sklearn 1.9 起 SVC(probability=True) 已弃用）
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('svm', CalibratedClassifierCV(
                SVC(kernel='rbf', C=C, gamma='scale', random_state=42),
                method='sigmoid', cv=5, ensemble=False)),
        ])
        self.is_trained = False
        self.model_info = {}

    # ---------- 数据集 ----------
    def load_dataset(self, data_dir, use_face_detect=True):
        """按文件夹名当标签加载数据集：data_dir/male/*.jpg -> 0，data_dir/female/*.jpg -> 1"""
        features, labels, file_paths = [], [], []
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 目录 {class_dir} 不存在")
                continue

            img_files = sorted(
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            )
            print(f"加载类别: {class_name}（{len(img_files)} 张）")
            no_face = 0
            for img_file in img_files:
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  跳过无法读取的图片: {img_file}")
                    continue
                if use_face_detect and detect_face_box(img) is None:
                    no_face += 1
                features.append(extract_features(img, use_face_detect))
                labels.append(label)
                file_paths.append(img_path)
            if no_face:
                print(f"  提示: 有 {no_face} 张没检出人脸，已退回整图提取特征")

        return np.array(features), np.array(labels), file_paths

    # ---------- 训练 ----------
    def train(self, train_dir=TRAIN_DIR, test_dir=TEST_DIR):
        print("=" * 60)
        print("加载训练数据...")
        X_train, y_train, _ = self.load_dataset(train_dir)
        if len(X_train) == 0:
            print("错误: 没有找到训练数据！")
            print("请先运行 build_dataset_from_utkface.py 生成数据集，")
            print("或者运行 collect_gender_data.py 用摄像头拍自己的照片。")
            return False
        print(f"训练集: {X_train.shape[0]} 张图片，特征维度 {X_train.shape[1]}")

        print("加载测试数据...")
        X_test, y_test, _ = self.load_dataset(test_dir)
        print(f"测试集: {X_test.shape[0]} 张图片")

        print("训练模型中（标准化 -> PCA -> SVM）...")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        print(f"训练耗时: {time.time() - t0:.1f} 秒")

        # 训练集准确率
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"训练集准确率: {train_accuracy:.3f}")

        # 5 折交叉验证（比单次划分更能反映真实水平）
        cv_accuracy = None
        if len(X_train) >= 100:
            try:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(self.model, X_train, y_train, cv=5, n_jobs=-1)
                cv_accuracy = float(scores.mean())
                print(f"5折交叉验证准确率: {cv_accuracy:.3f} "
                      f"(各折 {', '.join(f'{s:.3f}' for s in scores)})")
            except Exception as e:
                print(f"（交叉验证跳过：{e}）")

        # 测试集准确率 + 分类报告
        test_accuracy = None
        if len(X_test) > 0:
            test_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)
            print(f"测试集准确率: {test_accuracy:.3f}")
            print("\n分类报告:")
            print(classification_report(
                y_test, test_pred,
                target_names=[f"{c}({CLASS_NAMES_CN[c]})" for c in self.class_names],
                zero_division=0
            ))
            self.plot_confusion_matrix(y_test, test_pred)

        # 保存模型（joblib 存模型 + pickle 存模型信息）
        joblib.dump(self.model, MODEL_PATH)
        self.model_info = {
            'class_names': self.class_names,
            'class_names_cn': CLASS_NAMES_CN,
            'feature_dim': int(X_train.shape[1]),
            'img_size': IMG_SIZE,
            'pix_size': PIX_SIZE,
            'hist_bins': HIST_BINS,
            'use_face_detect': True,
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'train_accuracy': float(train_accuracy),
            'cv_accuracy': cv_accuracy,
            'test_accuracy': None if test_accuracy is None else float(test_accuracy),
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(INFO_PATH, 'wb') as f:
            pickle.dump(self.model_info, f)

        self.is_trained = True
        print(f"\n模型训练完成! 已保存为 {MODEL_PATH}")
        return True

    # ---------- 评估图 ----------
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        cn_labels = [CLASS_NAMES_CN[c] for c in self.class_names]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(cn_labels)), labels=cn_labels)
        ax.set_yticks(range(len(cn_labels)), labels=cn_labels)
        ax.set_xlabel('预测性别')
        ax.set_ylabel('真实性别')
        ax.set_title('性别识别混淆矩阵（测试集）')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        save_and_show(fig, 'confusion_matrix.png')

    def plot_train_samples(self, data_dir=TRAIN_DIR, n=12):
        """展示训练集样本"""
        rows = []
        for class_name in self.class_names:
            files = sorted(glob.glob(os.path.join(data_dir, class_name, '*')))
            rows.append((class_name, files[: n // 2]))

        fig, axes = plt.subplots(2, max(1, n // 2), figsize=(2 * (n // 2), 5))
        for r, (class_name, files) in enumerate(rows):
            for c in range(n // 2):
                ax = axes[r, c] if n // 2 > 1 else axes[r]
                ax.axis('off')
                if c < len(files):
                    img = cv2.imread(files[c])
                    if img is not None:
                        # 展示抠脸后的图，和训练时模型实际看到的输入一致
                        face, ok = crop_face(img)
                        ax.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        ax.set_title(f"{class_name}({CLASS_NAMES_CN[class_name]})"
                                     f"{'' if ok else ' 未检出'}", fontsize=9)
        fig.suptitle('训练集样本展示', fontsize=14)
        fig.tight_layout()
        save_and_show(fig, 'train_samples.png')

    # ---------- 预测 ----------
    def load_model(self):
        """加载训练好的模型"""
        if not (os.path.exists(MODEL_PATH) and os.path.exists(INFO_PATH)):
            return False
        self.model = joblib.load(MODEL_PATH)
        with open(INFO_PATH, 'rb') as f:
            self.model_info = pickle.load(f)
        self.class_names = self.model_info.get('class_names', CLASS_NAMES)
        self.is_trained = True
        return True

    def predict_image(self, image, use_face_detect=True):
        """
        预测单张图片（可以是摄像头整帧，也可以是已经裁好的人脸）。
        返回 dict：类别、中文类别、置信度、各类概率、人脸框
        """
        if not self.is_trained and not self.load_model():
            print("错误: 请先训练模型（python gender_classifier.py）!")
            return None

        # 只检测一次人脸：框既用于显示，也用于抠脸，避免 extract_features 里再检测一遍
        box = detect_face_box(image) if use_face_detect else None
        face, _ = crop_face(image, box) if use_face_detect else (image, False)
        feature = features_from_face(face)
        # 只调一次 predict_proba：类别用 argmax 推出来。
        # 若 predict 和 predict_proba 各调一次，PCA 变换和 SVM 决策会重复计算一遍（实测每帧多花约 12 ms）；
        # 另外要传二维数组而不是 list，sklearn 对 list 输入还要多做一次校验和转换（每帧多花约 5 ms）
        probabilities = self.model.predict_proba(feature.reshape(1, -1))[0]
        prediction = int(np.argmax(probabilities))

        return {
            'class': self.class_names[prediction],
            'class_cn': CLASS_NAMES_CN.get(self.class_names[prediction], '?'),
            'confidence': float(probabilities[prediction]),
            'all_probabilities': {
                self.class_names[i]: float(p)
                for i, p in enumerate(probabilities)
            },
            'face_box': None if box is None else tuple(int(v) for v in box),
        }


# ===================== 4. 工具函数 =====================
def save_and_show(fig, filename):
    """保存图片到 results/ 后显示（没有图形界面时也不会报错）"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path, dpi=120)
    print(f"图已保存: {path}")
    if os.environ.get('GENDER_NO_SHOW') != '1':
        try:
            plt.show()
        except Exception as e:      # 无显示环境（比如远程/无桌面）时跳过
            print(f"（无法弹窗显示图片：{e}）")
    plt.close(fig)


# ===================== 5. 主流程 =====================
def main():
    classifier = GenderClassifier()

    # 先看看数据长什么样
    if os.path.exists(os.path.join(TRAIN_DIR, CLASS_NAMES[0])):
        classifier.plot_train_samples()

    if classifier.train():
        print("\n训练成功完成!")

        # 每类挑几张测试图片实测一下
        print("\n单张图片预测:")
        for class_name in classifier.class_names:
            class_dir = os.path.join(TEST_DIR, class_name)
            if not os.path.exists(class_dir):
                continue
            test_files = sorted(
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )[:3]
            for test_file in test_files:
                img_path = os.path.join(class_dir, test_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                result = classifier.predict_image(img)
                if result:
                    print(f"  图片: {test_file}")
                    print(f"  真实类别: {class_name}({CLASS_NAMES_CN[class_name]})")
                    print(f"  预测类别: {result['class']}({result['class_cn']})")
                    print(f"  置信度: {result['confidence']:.3f}")
                    print("  ---")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
备选方案：用 TensorFlow/Keras 卷积神经网络做性别识别

这是一个基于 TensorFlow/Keras 的卷积神经网络方案，做性别 2 分类：male / female。
和 SVM 版（gender_classifier.py）用的是**同一份数据**（data/train、data/val、data/test），
两个方案可以直接对比准确率。

需要先装 tensorflow：   pip install tensorflow
（只装了 sklearn 的话跑 SVM 版就够了，这个文件是可选的）

运行： python CNN_gender.py

注意：MobileNetV2 的 imagenet 预训练权重需要联网到 storage.googleapis.com 下载，
      国内网络下不来也没关系，脚本会自动退化成「从零训练」。
"""

# ===================== 核心修复：解决Matplotlib/Seaborn中文显示 =====================
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',      # Windows 优先
    'PingFang SC',          # macOS 优先
    'WenQuanYi Micro Hei',  # Linux 优先
    'DejaVu Sans'           # 兜底字体
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# ===================== 1. 配置参数 =====================
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 2
# 注意：flow_from_directory 按文件夹名字母序排类别，female 在 male 前面
CLASS_NAMES = ['female', 'male']
CLASS_NAMES_CN = {'male': '男', 'female': '女'}

# 数据集路径（和 SVM 版共用同一份数据）
TRAIN_DIR = os.path.join('data', 'train')
VAL_DIR = os.path.join('data', 'val')
TEST_DIR = os.path.join('data', 'test')

MODEL_PATH = 'best_gender_model.h5'
RESULT_DIR = 'results'

# ===================== 2. 数据加载 =====================
# 人脸是左右近似对称的，所以这里打开 horizontal_flip
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True,
    seed=42
)
val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    seed=42
)
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    seed=42
)
print("类别索引映射：", train_generator.class_indices)


def save_and_show(fig, filename):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path, dpi=120)
    print(f"图已保存: {path}")
    if os.environ.get('GENDER_NO_SHOW') != '1':
        try:
            plt.show()
        except Exception as e:
            print(f"（无法弹窗显示图片：{e}）")
    plt.close(fig)


def plot_train_samples():
    """可视化训练集样本"""
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    batch_images, batch_labels = next(iter(train_generator))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(batch_images[i])
        name = CLASS_NAMES[int(np.argmax(batch_labels[i]))]
        ax.set_title(f"{name}（{CLASS_NAMES_CN[name]}）")
        ax.axis('off')
    fig.suptitle('训练集样本展示', fontsize=16)
    fig.tight_layout()
    save_and_show(fig, 'cnn_train_samples.png')


plot_train_samples()


# ===================== 3. 构建模型 =====================
def build_basic_cnn():
    """方案1：轻量 CNN（从零训练）"""
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu',
               input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_transfer_learning_model():
    """方案2：迁移学习（MobileNetV2 预训练权重 + 只训练顶层）"""
    try:
        base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS),
                                 include_top=False, weights='imagenet', alpha=0.75)
        print("已加载 MobileNetV2 imagenet 预训练权重")
    except Exception as e:
        print(f"下载预训练权重失败（{type(e).__name__}），改为从零训练 MobileNetV2")
        print("  想用预训练权重的话，可以手动下载 mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_no_top.h5")
        print("  放到 C:\\Users\\<用户名>\\.keras\\models\\ 目录下再重跑")
        base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS),
                                 include_top=False, weights=None, alpha=0.75)

    base_model.trainable = False   # 冻结基础层（先训练顶层）

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 选择模型：样本量不大时推荐迁移学习；没网/没权重的话把下面两行对调，用 build_basic_cnn()
model = build_transfer_learning_model()
# model = build_basic_cnn()

model.summary()

# ===================== 4. 训练模型 =====================
callbacks = [
    # 早停：验证集损失 8 轮不下降则停止，恢复最佳权重
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
    # 保存最佳模型
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
    # 学习率衰减
    ReduceLROnPlateau(factor=0.2, patience=3, monitor='val_loss', min_lr=1e-6)
]

steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, val_generator.samples // BATCH_SIZE)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)


def plot_training_curves(history):
    """训练曲线（全中文标签）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    ax1.set_title('训练/验证准确率', fontsize=14)
    ax1.set_xlabel('训练轮次（Epoch）', fontsize=12)
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='训练损失', linewidth=2)
    ax2.plot(history.history['val_loss'], label='验证损失', linewidth=2)
    ax2.set_title('训练/验证损失', fontsize=14)
    ax2.set_xlabel('训练轮次（Epoch）', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_and_show(fig, 'cnn_training_curves.png')


plot_training_curves(history)

# ===================== 5. 模型评估 =====================
best_model = load_model(MODEL_PATH)

test_loss, test_acc = best_model.evaluate(test_generator, verbose=0)
print(f"\n测试集最终评估结果：")
print(f"测试集损失：{test_loss:.4f}")
print(f"测试集准确率：{test_acc:.4f}")

test_pred = best_model.predict(test_generator, verbose=0)
test_pred_classes = np.argmax(test_pred, axis=1)
test_true_classes = test_generator.classes

print("\n测试集分类报告：")
print(classification_report(
    test_true_classes, test_pred_classes,
    target_names=[f"{c}({CLASS_NAMES_CN[c]})" for c in CLASS_NAMES],
    zero_division=0
))


def plot_confusion_matrix():
    cm = confusion_matrix(test_true_classes, test_pred_classes,
                          labels=list(range(NUM_CLASSES)))
    cn_labels = [CLASS_NAMES_CN[c] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES), labels=cn_labels)
    ax.set_yticks(range(NUM_CLASSES), labels=cn_labels)
    ax.set_xlabel('预测性别', fontsize=12)
    ax.set_ylabel('真实性别', fontsize=12)
    ax.set_title('性别识别混淆矩阵（测试集，CNN）', fontsize=14)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_and_show(fig, 'cnn_confusion_matrix.png')


plot_confusion_matrix()


# ===================== 6. 单张图片预测 =====================
def predict_single_gender(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print("图片加载失败，请检查路径！")
        return None, 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    pred_probs = best_model.predict(np.expand_dims(img_resized / 255.0, axis=0), verbose=0)[0]
    idx = int(np.argmax(pred_probs))
    name = CLASS_NAMES[idx]

    print(f"预测性别：{CLASS_NAMES_CN[name]}（{name}）  置信度：{pred_probs[idx] * 100:.2f}%")
    for i, prob in enumerate(pred_probs):
        print(f"  {CLASS_NAMES_CN[CLASS_NAMES[i]]}：{prob * 100:.2f}%")
    return name, float(pred_probs[idx])


# 示例：predict_single_gender('data/test/female/female_0000.jpg')

# ===================== 7. 迁移学习微调（可选） =====================
def fine_tune_model():
    m = load_model(MODEL_PATH)
    base_model = m.layers[0]
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 20   # 只训练最后 20 层
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    m.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

    history_fine = m.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS + 20,
        initial_epoch=history.epoch[-1],
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    m.save('fine_tuned_gender_model.h5')
    plot_training_curves(history_fine)
    return history_fine


# 精度不够时再打开：
# fine_tune_history = fine_tune_model()

# 基于机器视觉的人体性别识别

用传统机器视觉（OpenCV + scikit-learn）从图片或摄像头画面中判断人脸性别（男 / 女），并给出置信度与各类概率。主方案不需要 TensorFlow/PyTorch，在普通 CPU 上即可完成训练与实时推理。

---

## 一、项目解决什么问题

给定一张包含人脸的图片（或摄像头的一帧画面），回答两个问题：

1. **画面里有没有人脸？在什么位置？**
2. **这张脸的性别是男还是女？判断有多可信？**

实践中如果直接把整张图送进分类器，背景、衣服、光照都会干扰结果，所以本项目先把人脸检测出来、抠出人脸区域，再对人脸提取特征并分类。整个流程包括数据构建、模型训练、实时推理三个环节，适合作为机器视觉 / 机器学习的教学与实践项目，也适合作为传统特征工程 + SVM 方案的参考实现。

**适用边界：**

- 训练数据来自 UTKFace 中 16~70 岁的成年人脸，对儿童、严重遮挡、侧脸、极端光照的识别效果会下降；
- 只分析画面中面积最大的那张人脸，不做多人同时识别；
- 性别按 male / female 两类划分，不涉及其他性别认同维度；
- 精度约 92%（见「模型指标」），属于教学实验水平，**不适用于安防、招聘、身份核验等严肃决策场景**；
- **使用限制：不得用于身份认定、监控，或对个人进行未经同意的推断。**

---

## 二、主要功能

| 功能 | 对应文件 | 说明 |
| --- | --- | --- |
| 数据集构建 | `build_dataset_from_utkface.py` | 从 UTKFace 压缩包直接读取图片，筛选 16~70 岁样本，按种族分层抽样避免「肤色→性别」的捷径学习，按 8:1:1 划分为 train/val/test，并输出清单 CSV |
| 摄像头数据采集 | `collect_gender_data.py` | 按 `m`/`f` 键连续拍摄男/女人脸，每 5 张中 4 张进训练集、1 张进测试集；默认没检测到人脸就不保存 |
| 特征提取与训练 | `gender_classifier.py` | Haar 人脸检测 → 抠脸外扩 20% → 提取 2884 维特征 → 标准化 → PCA → RBF-SVM，保存整条流水线为 pkl |
| 实时性别识别 | `realtime_gender_detection.py` | 摄像头实时模式，画面显示人脸框、性别、置信度、各类概率、FPS；支持单图预测与文件夹批量测试 |
| 中文结果显示 | `realtime_gender_detection.py` | 用 PIL 在画面上绘制中文（`cv2.putText` 不支持中文），窗口标题在 Windows 上通过 pywin32 改为中文 |
| CNN 备选方案 | `CNN_gender.py` | TensorFlow/Keras + MobileNetV2 迁移学习的 2 分类实现，与 SVM 方案共用同一份数据，可对比准确率 |

**主方案技术细节：**

- **人脸检测**：OpenCV Haar 级联 `haarcascade_frontalface_default.xml`，直方图均衡后检测，取面积最大的框；
- **特征（共 2884 维）**：
  - HOG：64×64 窗口，16×16 块，8×8 步长，8×8 胞元，9 个方向 → 1764 维；
  - 32×32 灰度缩略图 → 1024 维；
  - HSV 三通道颜色直方图（每通道 32 bin）→ 96 维；
- **分类器**：`StandardScaler → PCA(300) → SVC(RBF, C=3, gamma='scale')`，用 `CalibratedClassifierCV`（sigmoid，5 折）校准概率；
- **模型产物**：`gender_classifier_model.pkl`（joblib）+ `model_info.pkl`（类别名、特征维度、准确率等元信息）。

**模型指标（仓库内已训练模型，测试集 400 张）：**

| 指标 | 数值 |
| --- | --- |
| 训练集准确率 | 99.75%（1600 张） |
| 5 折交叉验证准确率 | 91.1% |
| 测试集准确率 | **92.25%**（男 183/200 = 91.5%，女 186/200 = 93.0%） |
| 训练时间 | 2026-10-13 19:51:41 |

> 训练集准确率明显高于测试集，说明模型存在一定过拟合，这也是可进一步改进的方向。

---

## 三、项目结构

```text
第四节课/
├── gender_classifier.py           # 主方案：人脸检测、特征提取、SVM 训练与预测
├── realtime_gender_detection.py   # 实时 / 单图 / 批量三种推理模式
├── collect_gender_data.py         # 用摄像头采集自己的训练数据
├── build_dataset_from_utkface.py  # 从 UTKFace 构建 data/{train,val,test}
├── CNN_gender.py                  # 可选：TensorFlow 迁移学习方案
├── requirements.txt               # 主流程与可选依赖
├── .gitignore                     # 排除虚拟环境、数据集和生成物
├── package_clean.py               # 白名单打包脚本（排除隐私数据与生成物）
├── gender_classifier_model.pkl    # 已训练好的 SVM 流水线
├── model_info.pkl                 # 模型元信息（类别、特征维度、准确率等）
├── data/
│   ├── train/{male,female}/       # 训练集：每类 800 张
│   ├── val/{male,female}/         # 验证集：每类 200 张（仅 CNN 方案使用）
│   ├── test/{male,female}/        # 测试集：每类 200 张
│   └── dataset_manifest.csv       # 每张图的年龄、种族、来源文件名
├── _dataset_raw/
│   ├── utk_faces_images.zip       # UTKFace 原始图片（约 127 MB）
│   └── utk_dataset_metadata.csv   # UTKFace 标注（age / gender / race）
├── results/                       # 训练曲线、样本图、混淆矩阵
└── README.md
```

---

## 四、安装方法

### 4.1 环境要求

- Python 3.12+（本项目在 **Python 3.12.10** 下验证；使用更低版本时 pip 会自动选择兼容的依赖版本）
- Windows / macOS / Linux 均可；实时模式需要摄像头，其余模式不需要
- **必须使用 OpenCV 4.x**：OpenCV 5.0 起移除了 Haar 级联（`cv2.CascadeClassifier` 不存在），本项目的人脸检测无法运行。两个采集/识别脚本都会在启动时检查并给出提示。

### 4.2 创建虚拟环境并安装依赖

在项目根目录执行：

```bash
python -m venv .venv

# Windows（cmd / PowerShell）
.venv\Scripts\activate
# Windows（Git Bash）
source .venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate
```

安装主流程依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 已固定为下表验证过的版本；也可以手动安装：

```bash
pip install "opencv-python==4.11.0.86" numpy scikit-learn joblib matplotlib pillow
```

本项目验证时使用的版本为：opencv-python 4.11.0.86、numpy 2.5.4、scikit-learn 1.9.1、joblib 1.6.0、matplotlib 3.11.2、pillow 12.3.0（本地 `.venv` 已被 `.gitignore` 排除，请按 `requirements.txt` 创建自己的环境）。

按需安装可选依赖：

```bash
pip install pandas          # 构建数据集时
pip install pywin32         # 仅 Windows：中文窗口标题
pip install tensorflow      # 需要运行 CNN_gender.py 时
```

### 4.3 数据与模型

- 仓库中包含训练好的 `gender_classifier_model.pkl`、`model_info.pkl`，**只想体验识别的话可以直接跳到「使用方法」**，无需训练。
- `data/`、`_dataset_raw/` 和 `results/` 未随仓库提供（人脸属于敏感生物识别信息，且体积较大），需要按 5.1 节构建数据集。
- 重建数据集前，需将 UTKFace 的图片压缩包和标注 CSV 下载到 `_dataset_raw/`（脚本直接从 zip 读取，无需解压）：
  - `https://hf-mirror.com/datasets/nlphuji/utk_faces/resolve/main/utk_faces_images.zip`
  - `https://hf-mirror.com/datasets/nlphuji/utk_faces/resolve/main/utk_dataset_metadata.csv`
- 中文显示依赖系统字体：Windows 使用微软雅黑/黑体，macOS 使用苹方，Linux 建议安装文泉驿微米黑。缺少字体时图表中文可能显示为方框，但不影响识别功能。

---

## 五、使用方法

> 所有命令都在**项目根目录**下执行。

### 5.1 准备数据集（三选一）

**方式 A：使用本地已有数据**

若你本地已经按方式 B 或 C 准备好 `data/train`、`data/val`、`data/test`，可直接训练；仓库本身不包含数据集。

**方式 B：从 UTKFace 重新构建**

```bash
python build_dataset_from_utkface.py
```

默认每类：训练 800 张、验证 200 张、测试 200 张（合计 2400 张）。常用参数：

```bash
python build_dataset_from_utkface.py --force                      # 清空 data/ 下的旧图片后重建
python build_dataset_from_utkface.py --train-per-class 400 \
                                    --val-per-class 60 \
                                    --test-per-class 100          # 自定义每类数量
```

脚本会做三件事：只保留 16~70 岁样本；按种族分层抽样；把三个 split 互不重叠地切分。生成的 `data/dataset_manifest.csv` 记录每张图的年龄、种族和来源文件名。

**方式 C：用摄像头采集自己的照片**

```bash
python collect_gender_data.py
```

| 参数 | 说明 |
| --- | --- |
| `--camera 0` | 摄像头编号，默认 0 |
| `--every 0.5` | 自动拍照间隔（秒） |
| `--no-require-face` | 没检测到人脸也保存（默认不保存） |
| `--data-dir data` | 数据集根目录 |

操作按键：`m` 录男性、`f` 录女性、`s` 暂停、`q` 退出。采集时每 5 张中 1 张进 `data/test`，其余进 `data/train`；不会覆盖已有文件编号。

> 注意：SVM 主方案训练时只读取 `data/train` 和 `data/test`；`data/val` 仅供 `CNN_gender.py` 使用。

### 5.2 训练模型

```bash
python gender_classifier.py
```

执行流程：展示训练样本 → 加载 train/test → 训练 → 输出训练集准确率、5 折交叉验证准确率、测试集准确率与分类报告 → 保存模型。

**输出文件：**

- `gender_classifier_model.pkl`：完整流水线（标准化 + PCA + 概率校准 SVM）
- `model_info.pkl`：类别名、特征维度、样本数、各项准确率、训练时间
- `results/train_samples.png`：训练集样本（含实际抠脸效果）
- `results/confusion_matrix.png`：测试集混淆矩阵

没有图形界面时，设置环境变量跳过弹窗（图片仍会保存）：

```bash
# Git Bash / macOS / Linux
export GENDER_NO_SHOW=1
# Windows cmd
set GENDER_NO_SHOW=1
# PowerShell
$env:GENDER_NO_SHOW=1
```

### 5.3 实时识别 / 单图 / 批量测试

```bash
# 1) 打开默认摄像头，实时识别；按 q 或 ESC 退出，按 s 截图
python realtime_gender_detection.py

# 2) 指定摄像头编号，最多跑 300 帧后自动退出
python realtime_gender_detection.py --camera 1 --frames 300

# 3) 只测一张图片（会弹窗显示结果）
python realtime_gender_detection.py --image path/to/photo.jpg

# 4) 批量测试一个文件夹；文件夹名（male/female）作为真实标签，直接输出准确率
python realtime_gender_detection.py --dir data/test/female

# 5) 不弹窗，只打印结果（适合远程/无桌面环境）
python realtime_gender_detection.py --image path/to/photo.jpg --no-show
```

**摄像头画面的显示内容：**

- 人脸框：男性绿色，女性蓝色；
- 框上方：`男 male 89.8%` 形式的类别与置信度；
- 框下方：male / female 各自的概率；
- 左下角：当前处理帧率 `FPS xx.x`；
- 没检测到人脸时：显示“未检测到人脸，请把脸对准镜头”，不显示性别标签；
- 按 `s` 截图保存为当前目录下的 `gender_shot_YYYYmmdd_HHMMSS.jpg`。

**命令行参数：**

| 参数 | 说明 |
| --- | --- |
| `--image PATH` | 单张图片预测 |
| `--dir PATH` | 批量预测文件夹内所有图片，文件夹名作为真实标签 |
| `--camera N` | 摄像头编号（默认 0） |
| `--no-show` | 不创建窗口，只输出文本结果 |
| `--frames N` | 摄像头模式最多处理 N 帧，0 表示不限 |

启动时会先自动加载模型并打印训练时间与准确率；如果 `gender_classifier_model.pkl` 或 `model_info.pkl` 缺失，会提示先运行 `gender_classifier.py`。

### 5.4 可选：TensorFlow CNN 方案

```bash
python CNN_gender.py
```

- 使用 MobileNetV2 迁移学习（ImageNet 预训练权重，`alpha=0.75`），读取 `data/{train,val,test}`；
- 需要先安装 `tensorflow`；
- 预训练权重需要联网下载；下载失败会自动退化为从零训练，不影响脚本继续运行；
- 输出 `best_gender_model.h5`、`results/cnn_train_samples.png`、`results/cnn_training_curves.png`、`results/cnn_confusion_matrix.png`；
- 精度不足时可打开文件末尾注释掉的 `fine_tune_model()` 做微调，输出 `fine_tuned_gender_model.h5`。

### 5.5 打包干净文件

```bash
python package_clean.py
```

压缩包输出到 `dist/`，并附带 `MANIFEST.txt`（每个文件的大小与 SHA-256）。脚本采用白名单机制，只会打包源码、依赖清单、许可证、模型和文档；`.venv/`、`data/`、`_dataset_raw/`、`results/`、`__pycache__/` 等不会进入压缩包。

---

## 六、输入输出示例

### 6.1 单张图片

**输入**：任意包含正脸的照片，支持 jpg / jpeg / png / bmp；脚本对中文路径做了兼容处理（`np.fromfile + cv2.imdecode` 兜底）。

```bash
python realtime_gender_detection.py --image data/test/male/male_0000.jpg --no-show
```

**实际输出**：

```text
尝试加载模型...
✅ 模型加载成功! 训练时间 2026-10-13 19:51:41，训练集准确率 0.998，测试集准确率 0.922
图片: data/test/male/male_0000.jpg
预测性别: male（男）  置信度: 0.898
  male: 0.898
  female: 0.102
```

另一张测试图：

```text
图片: data/test/female/female_0000.jpg
预测性别: female（女）  置信度: 0.995
  male: 0.005
  female: 0.995
```

不加 `--no-show` 时会弹出一个窗口，在原图上画出人脸框并叠加中文标签。

### 6.2 批量测试文件夹

**输入**：一个以真实类别命名的文件夹，例如 `data/test/male/`。

```bash
python realtime_gender_detection.py --dir data/test/male --no-show
```

**实际输出（节选）**：

```text
批量测试 200 张图片，真实类别 = male
  √ male_0000.jpg  预测 male (0.898)
  ...
  √ male_0199.jpg  预测 male (0.910)
准确率: 183/200 = 0.915
```

在 `data/test/female` 上运行的实际结果：

```text
批量测试 200 张图片，真实类别 = female
  ...
准确率: 186/200 = 0.930
```

两组合计 369/400，即模型在测试集上的整体准确率为 **92.25%**，与 `model_info.pkl` 记录一致。

### 6.3 摄像头实时模式

**输入**：摄像头视频流（默认 0 号摄像头）。
**输出**：实时窗口画面 + 终端文本。`--no-show` 时终端每 30 帧打印一次：

```text
开始实时性别识别，按 q 退出，按 s 截图
[30] 男 94.2%
[60] 女 88.7%
```

### 6.4 作为 Python 模块调用

```python
import cv2
from gender_classifier import GenderClassifier

clf = GenderClassifier()
clf.load_model()                       # 加载 gender_classifier_model.pkl + model_info.pkl

img = cv2.imread('data/test/male/male_0000.jpg')
result = clf.predict_image(img)
print(result)
```

**实际返回的字典**：

```python
{
    'class': 'male',
    'class_cn': '男',
    'confidence': 0.8978,
    'all_probabilities': {'male': 0.8978, 'female': 0.1022},
    'face_box': (19, 4, 164, 164)      # (x, y, w, h)，未检出人脸时为 None
}
```

| 字段 | 含义 |
| --- | --- |
| `class` | 预测类别：`male` 或 `female` |
| `class_cn` | 中文类别：`男` / `女` |
| `confidence` | 预测类别的概率（0~1） |
| `all_probabilities` | 每个类别的概率 |
| `face_box` | 最大人脸框 `(x, y, w, h)`；未检出时为 `None` |

---

## 七、常见问题

**1. 启动时报 `cv2.CascadeClassifier` 不存在 / Haar 级联不可用**

说明当前环境是 OpenCV 5.x。安装回 4.x：

```bash
pip install "opencv-python==4.11.0.86"
```

**2. 运行时提示模型加载失败**

确认根目录下同时存在 `gender_classifier_model.pkl` 和 `model_info.pkl`；缺失或损坏时重新运行 `python gender_classifier.py` 训练。

**3. 无法打开摄像头**

检查摄像头是否被其他程序占用，或改用 `--camera 1` 尝试其他设备；也可以先用 `--image` 验证模型本身是否正常。

**4. 图表中文显示为方框**

系统缺少中文字体。Windows 安装微软雅黑、macOS 安装苹方即可；Linux 可安装 `fonts-wqy-microhei`。

**5. 检测不到人脸**

Haar 级联对侧脸、严重遮挡、过暗/过曝的图片较敏感。训练和推理时若未检出人脸，会退回用整张图提取特征，但这时结果不可靠；实时模式会直接提示“未检测到人脸”。

**6. 项目放在中文路径下**

代码已对两个典型问题做了处理：Haar 级联 XML 会复制到纯英文临时目录再加载；单图读取使用 Unicode 兼容写法。若仍遇到第三方依赖的路径问题，可把项目移到纯英文路径下。

---

## 八、说明

- 数据集来源：UTKFace（通过 `hf-mirror.com` 镜像获取），该数据集**仅限非商业研究用途**，图片版权归原始权利人所有，使用时请遵守 [UTKFace 官方条款](https://github.com/aicip/UTKFace)；本仓库不分发 `data/`、`_dataset_raw/`、`results/` 等含人脸的数据；
- 本项目仅用于技术学习与教学演示，性别识别结果不构成任何形式的身份认定；
- **使用限制：不得用于身份认定、监控，或对个人进行未经同意的推断。**

---

## 九、License 与模型授权说明

- **源代码**：采用 [Apache License 2.0](LICENSE) 授权。
- **模型文件 `gender_classifier_model.pkl`**：该模型基于 UTKFace 数据集训练，**仅供非商业研究与教学使用，不在 Apache 2.0 授权范围内**。UTKFace 的授权条款为 “available for non-commercial research purposes only”，本仓库不包含原始人脸图片，模型仅以研究/教学目的提供。
- **`model_info.pkl`**：仅包含类别、特征维度、准确率等模型元信息，随源代码一并按 Apache 2.0 提供。
- 第三方依赖遵循各自许可证（opencv-python、numpy、scikit-learn、joblib、pandas、matplotlib、Pillow、TensorFlow 等，均为宽松许可证），详见 [`NOTICE`](NOTICE)。
- **使用限制：本项目及模型不得用于身份认定、监控，或对个人进行未经同意的推断。**

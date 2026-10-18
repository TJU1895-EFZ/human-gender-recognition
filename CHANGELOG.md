# Changelog

## v1.0.0 (2026-10-18)

首个公开版本。

### 包含

- OpenCV Haar 人脸检测 + HOG / 灰度缩略图 / HSV 直方图特征 + PCA + RBF-SVM 的性别识别主流程
- 摄像头实时识别、单图识别、文件夹批量测试
- UTKFace 数据集构建脚本与摄像头数据采集脚本
- 可选 TensorFlow/Keras MobileNetV2 方案
- 已训练模型 `gender_classifier_model.pkl` 与模型元信息 `model_info.pkl`
- 干净打包脚本 `package_clean.py`（白名单 + SHA-256 清单）

### 说明

- 源代码采用 Apache License 2.0 授权，完整条款见 `LICENSE`。
- 模型基于 UTKFace 数据集训练，**仅供非商业研究与教学使用，不在 Apache License 2.0 授权范围内**。
- 仓库不包含 UTKFace 原始图片或任何采集到的人脸数据。
- **不得用于身份认定、监控，或对个人进行未经同意的推断。**

### 已知限制

- 训练数据为 UTKFace 中 16~70 岁成年人脸，对儿童、严重遮挡、侧脸和极端光照的效果会下降。
- 仅支持 male / female 两类，不涉及其他性别认同维度。
- 测试集准确率约 92.25%，属于教学实验水平，不适用于安防、招聘、身份核验等严肃决策场景。
- 项目未包含自动化测试与 CI。

<<<<<<< HEAD
# 基于深度学习的宠物狗情绪状态识别与异常行为预警系统

本项目面向毕业设计任务书要求，构建一个面向宠物狗视频的综合分析系统。系统支持上传宠物狗视频，同步完成情绪状态识别、异常行为检测、置信度展示和异常风险预警。

项目整合了两个子模块：

- `emotion`：宠物狗情绪状态识别，基于 ResNet50 完成五分类图像识别。
- `behavior`：宠物狗异常行为预警，基于视频运动特征和随机森林完成窗口级行为分类。

最终系统采用 Streamlit 构建前端，围绕“数据输入模块、模型推理模块、结果展示模块、异常预警模块、历史记录模块”组织功能流程。

## 1. 系统功能

| 模块 | 功能说明 |
| --- | --- |
| 数据输入模块 | 支持上传 `mp4`、`avi`、`mov`、`mkv` 格式视频，展示视频时长、帧率、分辨率、总帧数等基础信息。 |
| 模型推理模块 | 对上传视频抽取多帧图像进行情绪识别，同时按滑动窗口提取运动特征进行行为检测。 |
| 结果展示模块 | 展示情绪分类结果、行为分类结果、平均置信度、窗口级预测明细和推理耗时。 |
| 异常预警模块 | 根据异常窗口数量、异常窗口占比和主导行为类别输出低风险、中风险或高风险提示。 |
| 历史记录模块 | 自动保存每次成功推理的结果，支持按时间查阅历史情绪、行为、风险等级和窗口明细。 |

## 2. 技术路线

### 2.1 情绪识别

情绪识别模块位于 `emotion/`，使用 ResNet50 作为主干网络，对宠物狗图像进行五分类识别。

情绪类别包括：

| 标签 | 中文含义 |
| --- | --- |
| `happy` | 开心 |
| `alert` | 警觉/焦虑 |
| `angry` | 愤怒 |
| `frown` | 低落/恐惧 |
| `relax` | 放松 |

系统对视频进行多帧采样，每一帧单独输入情绪识别模型，再对各类别置信度求平均，得到视频级情绪识别结果。

### 2.2 行为预警

行为预警模块位于 `behavior/`，对视频进行滑动窗口切分，提取运动范围、速度、轨迹变化等统计特征，再使用随机森林模型进行分类。

行为类别包括：

| 标签 | 中文含义 | 是否异常 |
| --- | --- | --- |
| `normal` | 正常 | 否 |
| `long_static` | 长时间静止/无精打采 | 是 |
| `activity_drop` | 活动骤减 | 是 |
| `frequent_walking` | 频繁走动/焦虑徘徊 | 是 |

系统保留 `behavior/outputs_tuned` 作为最终行为模型输出目录，不再使用实验性的五分类版本。

## 3. 项目结构

```text
Petdog2.1/
├── app.py
├── README.md
├── requirements.txt
├── emotion/
│   ├── Dog Emotions - 5 Classes/
│   │   └── train_images_5_class/
│   ├── emotion_recognition/
│   │   ├── cli.py
│   │   ├── cnn_infer.py
│   │   ├── cnn_train.py
│   │   └── config.py
│   └── outputs_emotion/
│       ├── emotion_resnet50.pt
│       ├── emotion_cnn_summary.json
│       └── emotion_cnn_report.json
├── behavior/
│   ├── Activity Analysis/
│   ├── behavior_warning/
│   │   ├── cli.py
│   │   ├── features.py
│   │   ├── infer.py
│   │   ├── pseudo_labels.py
│   │   └── train.py
│   └── outputs_tuned/
│       ├── behavior_warning_model.joblib
│       ├── behavior_training_report.json
│       └── behavior_confusion_matrix.csv
└── integrated_outputs/
    └── behavior_inference/
```

## 4. 环境安装

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

核心依赖包括：

| 依赖 | 用途 |
| --- | --- |
| `streamlit` | 前端交互系统 |
| `torch`、`torchvision` | 情绪识别模型训练与推理 |
| `opencv-python` | 视频读取、抽帧和运动特征提取 |
| `pandas`、`numpy` | 数据处理 |
| `scikit-learn`、`joblib` | 行为分类模型训练、保存和加载 |

## 5. 运行系统

在项目根目录执行：

```bash
streamlit run app.py
```

系统默认加载以下模型：

```text
emotion/outputs_emotion/emotion_resnet50.pt
behavior/outputs_tuned/behavior_warning_model.joblib
```

进入页面后，操作流程为：

1. 在左侧侧边栏设置视频情绪采样帧数、行为检测窗口长度和窗口步长。
2. 在“视频识别与预警”页面上传宠物狗视频。
3. 检查视频基础信息和预览画面。
4. 点击“开始识别与预警”。
5. 查看情绪分类、行为分类、置信度、窗口级明细和异常风险提示。
6. 进入“历史记录”页面，查看已保存的历史推理结果。

## 6. 数据集说明

### 6.1 情绪图像数据

路径：

```text
emotion/Dog Emotions - 5 Classes/train_images_5_class
```

当前数据集共 5 类，每类 1865 张图像，共 9325 张图像，满足任务书“不低于 5000 张图像或 100 段视频”的样本规模要求。

### 6.2 行为视频数据

路径：

```text
behavior/Activity Analysis
```

当前包含 62 段宠物狗视频。行为模块通过滑动窗口生成窗口级训练样本，并使用规则生成伪标签用于监督训练。若后续需要严格满足“100 段视频”口径，可继续补充行为视频数据。

## 7. 模型训练与评估结果

### 7.1 情绪识别模型

模型文件：

```text
emotion/outputs_emotion/emotion_resnet50.pt
```

当前主要指标：

| 指标 | 结果 |
| --- | --- |
| 验证准确率 | 72.65% |
| Macro-F1 | 72.51% |
| 最佳 Epoch | 7 |

各类别表现中，`happy` 和 `relax` 较稳定，`alert`、`angry`、`frown` 存在一定混淆，论文中可作为后续优化方向说明。

### 7.2 行为预警模型

模型文件：

```text
behavior/outputs_tuned/behavior_warning_model.joblib
```

当前主要指标：

| 指标 | 结果 |
| --- | --- |
| Accuracy | 94.44% |
| Balanced Accuracy | 93.06% |
| Macro-F1 | 92.72% |
| Weighted-F1 | 94.44% |

各类别 F1：

| 类别 | F1 |
| --- | --- |
| `activity_drop` | 83.87% |
| `frequent_walking` | 95.04% |
| `long_static` | 96.86% |
| `normal` | 95.10% |

行为模型采用按视频分组的交叉验证，避免同一视频切分出的窗口同时进入训练集和测试集。

## 8. 评价指标

### 8.1 情绪识别模型

建议评价指标：

| 指标 | 说明 |
| --- | --- |
| Accuracy | 整体情绪分类正确率 |
| Macro-F1 | 五类情绪平均识别能力 |
| Precision | 某类被预测出来时的可靠程度 |
| Recall | 某类真实样本被识别出的比例 |
| 混淆矩阵 | 分析情绪类别之间的误判关系 |
| 推理耗时 | 评价视频抽帧识别是否满足系统响应要求 |

建议最终版标准：

- Accuracy 不低于 70%。
- Macro-F1 不低于 70%。
- 单类 F1 尽量不低于 60%。
- 视频情绪识别耗时尽量控制在 15 秒以内。

### 8.2 行为预警模型

建议评价指标：

| 指标 | 说明 |
| --- | --- |
| Accuracy | 整体行为分类正确率 |
| Balanced Accuracy | 类别不均衡情况下的平均识别能力 |
| Macro-F1 | 各行为类别的平均综合表现 |
| 异常类 Recall | 异常行为漏检情况 |
| 异常类 Precision | 异常预警误报情况 |
| 混淆矩阵 | 分析正常与异常行为之间的误判 |
| 预警响应时间 | 评价系统实际使用体验 |

建议最终版标准：

- Accuracy 不低于 85%。
- Balanced Accuracy 不低于 80%。
- Macro-F1 不低于 80%。
- 每类异常行为 Recall 尽量不低于 80%。
- 短视频行为预警耗时尽量控制在 30 秒以内。

## 9. 命令行训练与推理

### 9.1 训练情绪识别模型

```bash
cd emotion
python -m emotion_recognition.cli train --image-size 256 --batch-size 16 --freeze-backbone-epochs 3 --backbone-learning-rate-scale 0.2
```

### 9.2 单图情绪推理

```bash
cd emotion
python -m emotion_recognition.cli infer --image-path "your_image.jpg"
```

### 9.3 训练行为预警模型

```bash
cd behavior
python -m behavior_warning.cli pipeline --dataset-dir "Activity Analysis" --output-dir outputs_tuned --cv-folds 5 --window-seconds 12 --stride-seconds 4 --min-positive-windows 1 --min-video-zscore-windows 4 --long-static-z 0.7 --activity-drop-z 0.7 --frequent-walking-speed-z 0.6 --frequent-walking-distance-z 0.6
```

### 9.4 单视频行为推理

```bash
cd behavior
python -m behavior_warning.cli infer --video-path "Activity Analysis/your_video.mp4" --model-path outputs_tuned/behavior_warning_model.joblib --output-dir outputs_tuned/inference --window-seconds 12 --stride-seconds 4
```

## 10. 测试方案

建议从以下维度测试系统：

| 测试维度 | 测试内容 | 合格标准 |
| --- | --- | --- |
| 数据输入 | 上传常见视频格式、短视频、损坏视频 | 合法视频可读取，异常输入有提示 |
| 情绪识别 | 上传不同情绪状态视频 | 输出情绪类别、中文含义、平均置信度 |
| 行为预警 | 上传正常、长时间静止、活动骤减、频繁走动视频 | 输出行为类别、异常窗口占比和预警等级 |
| 响应速度 | 记录情绪识别和行为预警耗时 | 短视频整体响应在可接受范围内 |
| 稳定性 | 连续上传多个视频 | 页面不崩溃，模型路径不丢失 |
| 结果展示 | 检查指标卡片、表格、预警信息 | 信息完整、字段清晰 |

## 11. 参考项目优化点

参考 `yolov5_garbage_detect` 类视觉检测项目的组织方式，本项目做了以下整理：

- 将模型加载路径、推理流程、结果展示和运行方式集中到根目录 README。
- 前端页面以任务流程组织，而不是仅展示零散指标。
- 上传输入与推理按钮分离，避免上传后自动长时间运行。
- 保留模型指标、类别定义、预测置信度和预警结果，便于论文截图和测试报告整理。
- 删除重复说明文档，降低后续维护成本。

## 12. 毕业设计对应关系

| 任务书要求 | 项目实现 |
| --- | --- |
| 调研宠物行为识别、图像/视频分析、情绪识别技术 | README 中说明技术路线、类别定义和评价指标 |
| 制作宠物狗情绪与行为数据集 | 情绪图像 9325 张，行为视频 62 段，并生成窗口级行为样本 |
| 设计情绪识别与行为检测融合模型 | ResNet50 情绪识别 + 视频运动特征随机森林行为预警 |
| 构建完整系统 | Streamlit 前端包含上传、推理、结果展示、异常预警 |
| 设计测试方案并优化系统 | README 中给出测试维度、合格标准和当前模型指标 |

## 13. 后续优化方向

- 增加狗脸或狗体检测模块，在情绪识别前先裁剪有效区域。
- 扩充行为视频数量，补充更真实的异常行为样本。
- 对 `alert`、`angry`、`frown` 等易混淆情绪进行难样本增强。
- 引入时序深度模型，如 CNN-LSTM、SlowFast 或 Video Transformer，提升视频级行为理解能力。
- 将预警阈值改为可配置参数，便于不同场景调整敏感度。
=======
# petdog
>>>>>>> 78f907236e30f33d8c9664cfe9b5d3a31517b8c0

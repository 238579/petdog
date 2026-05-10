# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
EMOTION_ROOT = ROOT / "emotion"
BEHAVIOR_ROOT = ROOT / "behavior"

for package_root in (EMOTION_ROOT, BEHAVIOR_ROOT):
    package_path = str(package_root)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from behavior_warning.config import FeatureConfig  # noqa: E402
from behavior_warning.infer import infer_video  # noqa: E402
from emotion_recognition.cnn_infer import infer_emotion_cnn  # noqa: E402


EMOTION_MODEL = EMOTION_ROOT / "outputs_emotion" / "emotion_resnet50.pt"
BEHAVIOR_MODEL = BEHAVIOR_ROOT / "outputs_tuned" / "behavior_warning_model.joblib"
BEHAVIOR_OUTPUT = ROOT / "integrated_outputs" / "behavior_inference"
HISTORY_OUTPUT = ROOT / "integrated_outputs" / "history"
HISTORY_FILE = HISTORY_OUTPUT / "inference_history.jsonl"

EMOTION_NAMES = {
    "alert": "警觉/焦虑",
    "angry": "愤怒",
    "frown": "低落/恐惧",
    "happy": "开心",
    "relax": "放松",
}

BEHAVIOR_NAMES = {
    "normal": "正常",
    "long_static": "长时间静止/无精打采",
    "activity_drop": "活动骤减",
    "frequent_walking": "频繁走动/焦虑徘徊",
}

WARNING_TEXT = {
    "normal": "未检测到明显异常行为。",
    "long_static": "中风险：检测到长时间静止，建议观察精神状态、食欲和反应能力。",
    "activity_drop": "高风险：检测到活动骤减，建议及时关注健康状况。",
    "frequent_walking": "中风险：检测到频繁走动，可能存在焦虑、兴奋或环境压力。",
}


@dataclass
class BehaviorSummary:
    dominant_label: str
    dominant_confidence: float
    warning_windows: int
    total_windows: int
    warning_ratio: float
    timeline: pd.DataFrame
    elapsed_seconds: float


@dataclass
class EmotionVoteSummary:
    prediction: str
    confidence: float
    frame_count: int
    elapsed_seconds: float
    frame_results: pd.DataFrame
    score_results: pd.DataFrame


def risk_level(summary: BehaviorSummary) -> tuple[str, str]:
    if summary.warning_windows == 0 or summary.warning_ratio < 0.15:
        return "低风险", "未达到异常预警阈值，建议继续观察。"
    if summary.dominant_label == "activity_drop" or summary.warning_ratio >= 0.5:
        return "高风险", "异常窗口占比较高，建议尽快关注宠物健康状态。"
    return "中风险", "检测到连续或局部异常行为，建议结合宠物精神状态继续观察。"


def append_history(record: dict) -> None:
    HISTORY_OUTPUT.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []

    records = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        st.error(f"{label}不存在：{path}")
        st.stop()


def save_upload(uploaded_file, suffix: str) -> Path:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getbuffer())
    temp.flush()
    return Path(temp.name)


def read_video_metadata(video_path: Path) -> dict[str, str]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "视频时长": "无法读取",
            "帧率": "无法读取",
            "分辨率": "无法读取",
            "总帧数": "无法读取",
        }

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration = frame_count / fps if fps > 0 else 0
    return {
        "视频时长": f"{duration:.2f} 秒" if duration else "无法读取",
        "帧率": f"{fps:.2f} FPS" if fps else "无法读取",
        "分辨率": f"{width} x {height}" if width and height else "无法读取",
        "总帧数": str(frame_count) if frame_count else "无法读取",
    }


def extract_video_frames(video_path: Path, sample_count: int) -> List[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        positions = [0]
    else:
        positions = [
            int((idx + 1) * frame_count / (sample_count + 1))
            for idx in range(sample_count)
        ]

    frame_paths: List[Path] = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(position, 0))
        ok, frame = capture.read()
        if not ok:
            continue
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp.name, frame)
        frame_paths.append(Path(temp.name))

    capture.release()
    return frame_paths


def run_video_emotion_vote(video_path: Path, sample_count: int) -> Optional[EmotionVoteSummary]:
    frame_paths = extract_video_frames(video_path, sample_count)
    if not frame_paths:
        return None

    start = time.perf_counter()
    rows = []
    score_sum: Dict[str, float] = {}
    for index, frame_path in enumerate(frame_paths, start=1):
        result = infer_emotion_cnn(frame_path, EMOTION_MODEL)
        prediction = str(result["prediction"])
        scores = result.get("scores", {})
        confidence = float(scores.get(prediction, 0.0)) if isinstance(scores, dict) else 0.0
        rows.append(
            {
                "采样帧": index,
                "预测标签": prediction,
                "情绪类型": EMOTION_NAMES.get(prediction, prediction),
                "置信度": confidence,
            }
        )
        if isinstance(scores, dict):
            for label, score in scores.items():
                score_sum[label] = score_sum.get(label, 0.0) + float(score)

    averaged_scores = {
        label: score / max(len(frame_paths), 1)
        for label, score in score_sum.items()
    }
    prediction = max(averaged_scores, key=averaged_scores.get)
    score_rows = [
        {
            "英文标签": label,
            "中文含义": EMOTION_NAMES.get(label, label),
            "平均置信度": float(score),
        }
        for label, score in sorted(averaged_scores.items(), key=lambda item: item[1], reverse=True)
    ]
    return EmotionVoteSummary(
        prediction=prediction,
        confidence=float(averaged_scores[prediction]),
        frame_count=len(frame_paths),
        elapsed_seconds=time.perf_counter() - start,
        frame_results=pd.DataFrame(rows),
        score_results=pd.DataFrame(score_rows),
    )


def run_behavior(video_path: Path, window_seconds: int, stride_seconds: int) -> BehaviorSummary:
    start = time.perf_counter()
    output_csv = infer_video(
        video_path=video_path,
        model_path=BEHAVIOR_MODEL,
        output_dir=BEHAVIOR_OUTPUT,
        config=FeatureConfig(
            output_dir=BEHAVIOR_OUTPUT,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            sample_fps=2.0,
        ),
    )
    timeline = pd.read_csv(output_csv)
    if timeline.empty:
        raise RuntimeError("该视频未生成有效行为窗口，请换用更长或更清晰的视频。")

    labels = timeline["prediction"].astype(str)
    counts = labels.value_counts()
    dominant_label = str(counts.index[0])

    score_column = f"score_{dominant_label}"
    if score_column in timeline.columns:
        dominant_confidence = float(timeline.loc[labels == dominant_label, score_column].mean())
    else:
        dominant_confidence = float(counts.iloc[0] / len(timeline))

    warning_windows = int((labels != "normal").sum())
    total_windows = int(len(timeline))

    return BehaviorSummary(
        dominant_label=dominant_label,
        dominant_confidence=dominant_confidence,
        warning_windows=warning_windows,
        total_windows=total_windows,
        warning_ratio=warning_windows / max(total_windows, 1),
        timeline=timeline,
        elapsed_seconds=time.perf_counter() - start,
    )


def render_sidebar() -> tuple[int, int, int]:
    st.sidebar.title("系统配置")
    st.sidebar.caption("系统默认复用已训练完成的 ResNet50 情绪识别模型和行为预警模型。")
    sample_frames = st.sidebar.slider("视频情绪采样帧数", min_value=1, max_value=9, value=5, step=1)
    window_seconds = st.sidebar.slider("行为检测窗口长度（秒）", min_value=4, max_value=30, value=12, step=2)
    stride_seconds = st.sidebar.slider("行为检测步长（秒）", min_value=2, max_value=20, value=4, step=2)
    return sample_frames, window_seconds, stride_seconds


def render_metrics() -> None:
    emotion_summary = EMOTION_ROOT / "outputs_emotion" / "emotion_cnn_summary.json"
    behavior_report = BEHAVIOR_ROOT / "outputs_tuned" / "behavior_training_report.json"

    st.subheader("已训练模型指标")
    col_a, col_b, col_c, col_d = st.columns(4)

    if emotion_summary.exists():
        data = json.loads(emotion_summary.read_text(encoding="utf-8"))
        col_a.metric("情绪验证准确率", f"{float(data.get('best_val_accuracy', 0)):.2%}")
        col_b.metric("情绪 Macro-F1", f"{float(data.get('best_val_macro_f1', 0)):.2%}")

    if behavior_report.exists():
        data = json.loads(behavior_report.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        col_c.metric("行为准确率", f"{float(metrics.get('accuracy', 0)):.2%}")
        col_d.metric("行为 Macro-F1", f"{float(metrics.get('macro_f1', 0)):.2%}")


def show_video_emotion(summary: EmotionVoteSummary) -> None:
    st.markdown("#### 结果展示：情绪分类")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("视频情绪识别结果", EMOTION_NAMES.get(summary.prediction, summary.prediction))
    col_b.metric("平均置信度", f"{summary.confidence:.2%}")
    col_c.metric("情绪推理耗时", f"{summary.elapsed_seconds:.2f} 秒")
    st.progress(min(max(summary.confidence, 0.0), 1.0), text=f"视频情绪平均置信度：{summary.confidence:.2%}")
    st.write("情绪类别平均置信度")
    st.dataframe(summary.score_results, use_container_width=True, hide_index=True)
    st.write("采样帧识别明细")
    st.dataframe(summary.frame_results, use_container_width=True, hide_index=True)


def show_behavior_result(summary: BehaviorSummary) -> None:
    st.markdown("#### 结果展示：行为分类")
    label_name = BEHAVIOR_NAMES.get(summary.dominant_label, summary.dominant_label)
    warning_text = WARNING_TEXT.get(summary.dominant_label, "建议进一步观察。")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("主导行为", label_name)
    col_b.metric("主导置信度", f"{summary.dominant_confidence:.2%}")
    col_c.metric("异常窗口占比", f"{summary.warning_ratio:.2%}")
    col_d.metric("行为推理耗时", f"{summary.elapsed_seconds:.2f} 秒")

    st.markdown("#### 异常预警")
    level, advice = risk_level(summary)
    if level == "低风险":
        st.success(f"{level}：{warning_text} {advice}")
    elif level == "中风险":
        st.warning(f"{level}：{warning_text} {advice} 异常窗口：{summary.warning_windows}/{summary.total_windows}。")
    else:
        st.error(f"{level}：{warning_text} {advice} 异常窗口：{summary.warning_windows}/{summary.total_windows}。")

    display_columns = [
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "prediction",
    ]
    score_columns = [column for column in summary.timeline.columns if column.startswith("score_")]
    timeline = summary.timeline[
        [column for column in display_columns + score_columns if column in summary.timeline.columns]
    ].copy()
    if "prediction" in timeline.columns:
        timeline["预测类别"] = timeline["prediction"].map(
            lambda value: BEHAVIOR_NAMES.get(str(value), str(value))
        )
    st.dataframe(timeline, use_container_width=True, hide_index=True)


def build_history_record(
    uploaded_name: str,
    metadata: dict[str, str],
    sample_frames: int,
    window_seconds: int,
    stride_seconds: int,
    emotion_summary: Optional[EmotionVoteSummary],
    behavior_summary: BehaviorSummary,
) -> dict:
    level, advice = risk_level(behavior_summary)
    record = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_name": uploaded_name,
        "video_metadata": metadata,
        "config": {
            "sample_frames": sample_frames,
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
        },
        "emotion": None,
        "behavior": {
            "label": behavior_summary.dominant_label,
            "label_cn": BEHAVIOR_NAMES.get(behavior_summary.dominant_label, behavior_summary.dominant_label),
            "confidence": behavior_summary.dominant_confidence,
            "warning_windows": behavior_summary.warning_windows,
            "total_windows": behavior_summary.total_windows,
            "warning_ratio": behavior_summary.warning_ratio,
            "elapsed_seconds": behavior_summary.elapsed_seconds,
        },
        "risk": {
            "level": level,
            "advice": advice,
        },
        "behavior_timeline": behavior_summary.timeline.to_dict(orient="records"),
    }

    if emotion_summary is not None:
        record["emotion"] = {
            "label": emotion_summary.prediction,
            "label_cn": EMOTION_NAMES.get(emotion_summary.prediction, emotion_summary.prediction),
            "confidence": emotion_summary.confidence,
            "frame_count": emotion_summary.frame_count,
            "elapsed_seconds": emotion_summary.elapsed_seconds,
            "score_results": emotion_summary.score_results.to_dict(orient="records"),
            "frame_results": emotion_summary.frame_results.to_dict(orient="records"),
        }
    return record


def render_history_tab() -> None:
    st.subheader("历史记录模块")
    records = load_history()
    if not records:
        st.info("暂无历史推理记录。完成一次视频识别与预警后，结果会自动保存到这里。")
        return

    rows = []
    for index, record in enumerate(records):
        emotion = record.get("emotion") or {}
        behavior = record.get("behavior") or {}
        risk = record.get("risk") or {}
        rows.append(
            {
                "序号": index + 1,
                "推理时间": record.get("created_at", ""),
                "视频名称": record.get("video_name", ""),
                "情绪结果": emotion.get("label_cn", "未生成"),
                "情绪置信度": f"{float(emotion.get('confidence', 0)):.2%}" if emotion else "未生成",
                "行为结果": behavior.get("label_cn", ""),
                "行为置信度": f"{float(behavior.get('confidence', 0)):.2%}",
                "异常占比": f"{float(behavior.get('warning_ratio', 0)):.2%}",
                "风险等级": risk.get("level", ""),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    selected = st.selectbox(
        "选择一条历史记录查看详情",
        options=list(range(len(records))),
        format_func=lambda idx: f"{records[idx].get('created_at', '')} - {records[idx].get('video_name', '')}",
    )
    record = records[selected]
    emotion = record.get("emotion") or {}
    behavior = record.get("behavior") or {}
    risk = record.get("risk") or {}

    st.markdown("#### 历史结果详情")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("情绪分类", emotion.get("label_cn", "未生成"))
    col_b.metric("情绪置信度", f"{float(emotion.get('confidence', 0)):.2%}" if emotion else "未生成")
    col_c.metric("行为分类", behavior.get("label_cn", ""))
    col_d.metric("风险等级", risk.get("level", ""))

    st.write("视频信息")
    metadata = record.get("video_metadata") or {}
    st.dataframe(pd.DataFrame(metadata.items(), columns=["项目", "值"]), use_container_width=True, hide_index=True)

    st.write("推理配置")
    config = record.get("config") or {}
    st.dataframe(pd.DataFrame(config.items(), columns=["参数", "值"]), use_container_width=True, hide_index=True)

    if emotion.get("score_results"):
        st.write("情绪类别置信度")
        st.dataframe(pd.DataFrame(emotion["score_results"]), use_container_width=True, hide_index=True)

    timeline = record.get("behavior_timeline") or []
    if timeline:
        st.write("行为窗口明细")
        st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="宠物狗情绪识别与异常行为预警系统",
        page_icon=":dog:",
        layout="wide",
    )

    require_file(EMOTION_MODEL, "情绪识别模型")
    require_file(BEHAVIOR_MODEL, "行为预警模型")

    sample_frames, window_seconds, stride_seconds = render_sidebar()

    st.title("基于深度学习的宠物狗情绪状态识别与异常行为预警系统")
    st.caption("仅保留视频上传入口，完成视频情绪状态识别、异常行为检测、置信度展示、响应耗时统计与异常预警。")

    render_metrics()

    tab_video, tab_history = st.tabs(["视频识别与预警", "历史记录"])

    with tab_video:
        st.subheader("数据输入模块")
        uploaded_video = st.file_uploader(
            "上传宠物狗视频，系统将同步完成情绪状态识别与异常行为预警",
            type=["mp4", "avi", "mov", "mkv"],
            key="video",
        )
        if uploaded_video is not None:
            video_path = save_upload(uploaded_video, Path(uploaded_video.name).suffix or ".mp4")
            left, right = st.columns([1.2, 1])
            with left:
                st.video(str(video_path))
            with right:
                st.write("视频信息")
                metadata = read_video_metadata(video_path)
                st.dataframe(
                    pd.DataFrame(metadata.items(), columns=["项目", "值"]),
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("模型推理模块")
            st.write(
                f"当前配置：抽取 {sample_frames} 帧进行情绪投票；"
                f"行为窗口长度 {window_seconds} 秒，步长 {stride_seconds} 秒。"
            )

            if st.button("开始识别与预警", type="primary", use_container_width=True):
                emotion_summary = None
                with st.spinner("正在进行视频多帧情绪投票..."):
                    emotion_summary = run_video_emotion_vote(video_path, sample_frames)
                    if emotion_summary is not None:
                        show_video_emotion(emotion_summary)
                    else:
                        st.info("未能抽取视频帧，将仅进行行为预警分析。")

                with st.spinner("正在进行行为窗口分析与异常预警..."):
                    try:
                        behavior_summary = run_behavior(video_path, window_seconds, stride_seconds)
                        show_behavior_result(behavior_summary)
                        append_history(
                            build_history_record(
                                uploaded_name=uploaded_video.name,
                                metadata=metadata,
                                sample_frames=sample_frames,
                                window_seconds=window_seconds,
                                stride_seconds=stride_seconds,
                                emotion_summary=emotion_summary,
                                behavior_summary=behavior_summary,
                            )
                        )
                        st.success("本次推理结果已保存到历史记录。")
                    except Exception as exc:
                        st.error(f"行为预警失败：{exc}")

    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    main()

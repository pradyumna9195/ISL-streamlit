import json
import os
import threading
from pathlib import Path

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from inference import InferenceState, load_trained_model
from settings import (
    ACTIONS,
    DEFAULT_THRESHOLD,
    MAX_SENTENCE_LENGTH,
    MODEL_PATH,
    PREDICTION_STABILITY_WINDOW,
    PROB_COLORS,
    SEQUENCE_LENGTH,
)
from vision import (
    draw_sentence_banner,
    draw_styled_landmarks,
    extract_keypoints,
    mediapipe_detection,
    mp_holistic,
    prob_viz,
)


class ISLVideoProcessor(VideoProcessorBase):
    def __init__(self, model, threshold):
        self.model = model
        self.state = InferenceState(
            actions=ACTIONS,
            sequence_length=SEQUENCE_LENGTH,
            stability_window=PREDICTION_STABILITY_WINDOW,
            max_sentence_length=MAX_SENTENCE_LENGTH,
            threshold=threshold,
        )
        self.lock = threading.Lock()
        self.latest_action = None
        self.latest_confidence = 0.0
        self.latest_probabilities = np.zeros(len(ACTIONS), dtype=np.float32)
        self.last_error = None
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def set_threshold(self, threshold):
        with self.lock:
            self.state.update_threshold(threshold)

    def reset(self):
        with self.lock:
            self.state.reset()
            self.latest_action = None
            self.latest_confidence = 0.0
            self.latest_probabilities = np.zeros(len(ACTIONS), dtype=np.float32)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        try:
            image, results = mediapipe_detection(image, self.holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            has_gesture = bool(results.left_hand_landmarks or results.right_hand_landmarks)

            with self.lock:
                output = self.state.process(self.model, keypoints, has_gesture=has_gesture)
                if output["action"] is not None:
                    self.latest_action = output["action"]
                else:
                    self.latest_action = None
                self.latest_confidence = output["confidence"]

                probabilities = output["probabilities"]
                sentence = output["sentence"]
                if probabilities is not None:
                    self.latest_probabilities = probabilities
                else:
                    self.latest_probabilities = np.zeros(len(ACTIONS), dtype=np.float32)

            image = prob_viz(self.latest_probabilities, ACTIONS, image, PROB_COLORS)
            image = draw_sentence_banner(image, sentence)

        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def __del__(self):
        if hasattr(self, "holistic"):
            self.holistic.close()


@st.cache_resource
def get_model(model_path: str):
    return load_trained_model(model_path)


def get_rtc_configuration():
    ice_servers_json = os.getenv("ICE_SERVERS_JSON", "")
    if ice_servers_json:
        try:
            return {"iceServers": json.loads(ice_servers_json)}
        except json.JSONDecodeError:
            pass
    return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def main():
    st.set_page_config(
        page_title="ISL Real-Time Recognition",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #f8fbff;
        }
        [data-testid="stSidebar"] {
            background: #f3f8ff;
        }
        [data-testid="stHeader"] {
            background: #f8fbff;
        }
        [data-testid="stToolbar"] {
            right: 1rem;
        }
        .hospital-hero {
            background: linear-gradient(120deg, #0f6cbd 0%, #1f8f7a 100%);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 8px 24px rgba(16, 70, 112, 0.12);
        }
        .hospital-hero h1 {
            color: #ffffff !important;
            margin: 0;
            font-size: 1.55rem;
            letter-spacing: 0.2px;
        }
        .hospital-hero p {
            color: #eaf6ff !important;
            margin-top: 0.35rem;
            margin-bottom: 0;
            font-size: 0.96rem;
        }
        .hospital-meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.6rem;
            margin-bottom: 0.85rem;
        }
        .hospital-meta-card {
            background: #ffffff;
            border: 1px solid #d6e6f7;
            border-radius: 12px;
            padding: 0.55rem 0.8rem;
        }
        .hospital-meta-card .k {
            color: #376383;
            font-size: 0.76rem;
            margin-bottom: 0.16rem;
        }
        .hospital-meta-card .v {
            color: #0f355e;
            font-size: 0.92rem;
            font-weight: 600;
            line-height: 1.3;
        }
        h1, h2, h3 {
            color: #0f355e !important;
        }
        [data-testid="stCaptionContainer"] p,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #1f3b57 !important;
        }
        label, .stSlider label, .stSlider span {
            color: #234867 !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background-color: #0f6cbd;
            border-color: #0f6cbd;
        }
        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background: #d6e9ff;
        }
        .controls-card {
            background: #ffffff;
            border: 1px solid #d6e6f7;
            border-radius: 14px;
            padding: 0.85rem 1rem 0.95rem 1rem;
            margin-bottom: 0.9rem;
        }
        .controls-title {
            color: #0f4a77;
            font-size: 0.86rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.2rem;
        }
        .actions-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.4rem;
        }
        .action-pill {
            border: 1px solid #c7dff4;
            color: #1f4e72;
            background: #f3f9ff;
            border-radius: 999px;
            padding: 0.16rem 0.58rem;
            font-size: 0.78rem;
        }
        [data-testid="metric-container"] {
            background: #f7fbff;
            border: 1px solid #d8e7f7;
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
        }
        .stButton > button {
            border: 1px solid #0f6cbd;
            color: #0f6cbd;
            background: #eef6ff;
            border-radius: 8px;
        }
        .stButton > button:hover {
            border-color: #0b5ca3;
            color: #0b5ca3;
            background: #e3f1ff;
        }
        .prediction-panel-title {
            color: #0f4a77;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hospital-hero">
            <h1>🏥 ISL Clinical Communication Console</h1>
            <p>Real-time Indian Sign Language interpretation support for healthcare environments.</p>
        </div>
        <div class="hospital-meta">
            <div class="hospital-meta-card">
                <div class="k">Deployment Focus</div>
                <div class="v">Healthcare & Clinical Desk</div>
            </div>
            <div class="hospital-meta-card">
                <div class="k">Inference Mode</div>
                <div class="v">Live Video + LSTM Sequence</div>
            </div>
            <div class="hospital-meta-card">
                <div class="k">Current Theme</div>
                <div class="v">Hospital Light Interface</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not Path(MODEL_PATH).exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    try:
        model = get_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    st.markdown('<div class="controls-card"><div class="controls-title">Detection Controls</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.95,
        value=float(DEFAULT_THRESHOLD),
        step=0.05,
    )

    action_pills = "".join([f'<span class="action-pill">{action}</span>' for action in ACTIONS.tolist()])
    st.markdown(
        f"""
        <div class="controls-title" style="margin-top:0.35rem;">Supported actions</div>
        <div class="actions-row">{action_pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rtc_config = get_rtc_configuration()
    webrtc_ctx = webrtc_streamer(
        key="isl-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: ISLVideoProcessor(model=model, threshold=threshold),
        async_processing=True,
    )

    left, right = st.columns([2, 1])
    with right:
        with st.container(border=True):
            st.markdown('<div class="prediction-panel-title">Inference Snapshot</div>', unsafe_allow_html=True)
            st.subheader("Live prediction")
            reset_clicked = st.button("Reset sentence", use_container_width=True)

            if webrtc_ctx.video_processor:
                processor = webrtc_ctx.video_processor
                processor.set_threshold(threshold)
                if reset_clicked:
                    processor.reset()

                st.metric("Current action", processor.latest_action or "-")
                st.metric("Confidence", f"{processor.latest_confidence:.2f}")
                st.write("Sentence:", " ".join(processor.state.sentence) or "-")

                if processor.last_error:
                    st.warning(f"Processing warning: {processor.last_error}")
            else:
                st.info("Start the camera to begin real-time inference.")

    with left:
        st.subheader("Camera stream")
        st.write("Allow browser camera permissions when prompted.")


if __name__ == "__main__":
    main()

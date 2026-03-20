import json
import os
import threading
from pathlib import Path

import av
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

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        try:
            image, results = mediapipe_detection(image, self.holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)

            with self.lock:
                output = self.state.process(self.model, keypoints)
                if output["action"] is not None:
                    self.latest_action = output["action"]
                self.latest_confidence = output["confidence"]

                probabilities = output["probabilities"]
                sentence = output["sentence"]

            if probabilities is not None:
                image = prob_viz(probabilities, ACTIONS, image, PROB_COLORS)
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
    st.set_page_config(page_title="ISL Real-Time Recognition", layout="wide")
    st.title("ISL Real-Time Recognition")
    st.caption("Streamlit + WebRTC + MediaPipe + LSTM inference")

    if not Path(MODEL_PATH).exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    try:
        model = get_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    threshold = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.95,
        value=float(DEFAULT_THRESHOLD),
        step=0.05,
    )

    st.write("Actions:", ", ".join(ACTIONS.tolist()))

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
        st.subheader("Live prediction")
        reset_clicked = st.button("Reset sentence")

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

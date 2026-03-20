# ISL Streamlit Real-Time App

This project converts the notebook workflow in `isl1.ipynb` into a deployable real-time app using Streamlit + WebRTC.

## Features

- Browser camera input via `streamlit-webrtc`
- MediaPipe Holistic landmark detection
- 30-frame sequence buffering for LSTM inference
- Real-time action prediction and confidence visualization
- Sentence stabilization logic matching the notebook behavior

## Project Files

- `app.py` — Streamlit UI + WebRTC stream setup
- `vision.py` — MediaPipe + drawing + keypoint extraction helpers
- `inference.py` — model loading and sequence inference state
- `settings.py` — constants and model path
- `action_best.h5` — trained model weights

## Run Locally

1. Create and activate a Python environment (recommended Python 3.10/3.11).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown by Streamlit and allow camera access.

## Notes for Cloud Deployment

- Use HTTPS for browser camera support.
- For reliable remote WebRTC, configure STUN/TURN servers.
- If Streamlit Cloud shows `ImportError: libGL.so.1`, ensure `packages.txt` is present in repo root (this project includes it).
- On newer Debian images (e.g., trixie), use `libglib2.0-0t64` instead of `libglib2.0-0`.
- You can pass ICE server config through an env var:
  - `ICE_SERVERS_JSON` as JSON array, e.g.

    ```json
    [{ "urls": ["stun:stun.l.google.com:19302"] }]
    ```

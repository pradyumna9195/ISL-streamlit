import json
from dataclasses import dataclass, field

import h5py
import numpy as np
from tensorflow.keras.models import load_model, model_from_json


@dataclass
class InferenceState:
    actions: np.ndarray
    sequence_length: int
    stability_window: int
    max_sentence_length: int
    threshold: float
    sequence: list = field(default_factory=list)
    predictions: list = field(default_factory=list)
    sentence: list = field(default_factory=list)

    def reset(self):
        self.sequence.clear()
        self.predictions.clear()
        self.sentence.clear()

    def update_threshold(self, threshold: float):
        self.threshold = float(threshold)

    def process(self, model, keypoints: np.ndarray, has_gesture: bool = True):
        if not has_gesture:
            self.sequence.clear()
            self.predictions.clear()
            return {
                "ready": False,
                "confidence": 0.0,
                "action": None,
                "sentence": self.sentence,
                "probabilities": None,
            }

        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.sequence_length :]

        if len(self.sequence) < self.sequence_length:
            return {
                "ready": False,
                "confidence": 0.0,
                "action": None,
                "sentence": self.sentence,
                "probabilities": None,
            }

        result = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
        prediction_idx = int(np.argmax(result))
        confidence = float(result[prediction_idx])
        self.predictions.append(prediction_idx)

        if len(self.predictions) >= self.stability_window:
            recent = self.predictions[-self.stability_window :]
            if len(np.unique(recent)) == 1 and confidence > self.threshold:
                label = self.actions[prediction_idx]
                if not self.sentence or label != self.sentence[-1]:
                    self.sentence.append(label)

        if len(self.sentence) > self.max_sentence_length:
            self.sentence = self.sentence[-self.max_sentence_length :]

        return {
            "ready": True,
            "confidence": confidence,
            "action": self.actions[prediction_idx],
            "sentence": self.sentence,
            "probabilities": result,
        }


def load_trained_model(model_path):
    try:
        return load_model(model_path, compile=False)
    except Exception as original_error:
        try:
            with h5py.File(model_path, "r") as h5_file:
                raw_model_config = h5_file.attrs.get("model_config")
                if raw_model_config is None:
                    raise ValueError("Missing model_config in H5 file.")

                if isinstance(raw_model_config, bytes):
                    raw_model_config = raw_model_config.decode("utf-8")

                model_config = json.loads(raw_model_config)

            def _normalize_input_layer_config(obj):
                if isinstance(obj, dict):
                    normalized = {}
                    for key, value in obj.items():
                        normalized[key] = _normalize_input_layer_config(value)

                    class_name = normalized.get("class_name")
                    config = normalized.get("config")
                    if class_name == "InputLayer" and isinstance(config, dict):
                        if "batch_shape" in config and "batch_input_shape" not in config:
                            config["batch_input_shape"] = config.pop("batch_shape")
                    return normalized

                if isinstance(obj, list):
                    return [_normalize_input_layer_config(item) for item in obj]

                return obj

            normalized_config = _normalize_input_layer_config(model_config)
            model = model_from_json(json.dumps(normalized_config))
            model.load_weights(model_path)
            return model
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load model using default and legacy fallback loaders. "
                f"Default error: {original_error}. Fallback error: {fallback_error}"
            ) from fallback_error

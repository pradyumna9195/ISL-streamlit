from dataclasses import dataclass, field

import numpy as np
from tensorflow.keras.models import load_model


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
    return load_model(model_path)

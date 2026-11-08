"""Shared application state: database, trained model, reference profiles and settings."""
from __future__ import annotations

import uuid
from pathlib import Path

from .analyzer import FrameAnalyzer, ImageAnalyzer, VideoAnalyzer
from .classifier import MudraClassifier
from .database import Database
from .features import feature_matrix
from .hand_tracker import HandTracker
from .similarity import ReferenceProfiles


class AppContext:
    def __init__(self):
        self.db = Database()
        self.db.initialize()
        self.classifier = MudraClassifier()
        self.classifier.load()
        self.profiles = ReferenceProfiles()
        self.reload_profiles()

    def S(self) -> dict:
        s = self.db.get_settings()
        self.profiles.tau = s["sim_tau"]
        return s

    @property
    def user_id(self) -> int:
        return int(self.db.get_settings()["current_user_id"])

    def reload_profiles(self) -> None:
        self.profiles.rebuild(self.db)

    def reload_model(self) -> None:
        self.classifier.load()
        self.reload_profiles()

    def tracker(self, static: bool = False, max_hands: int = 2) -> HandTracker:
        s = self.S()
        return HandTracker(static, max_hands, s["det_conf"], s["track_conf"], s["model_complexity"], s)

    def frame_analyzer(self) -> FrameAnalyzer:
        return FrameAnalyzer(self.classifier, self.profiles, self.S())

    def video_analyzer(self) -> VideoAnalyzer:
        return VideoAnalyzer(self.classifier, self.profiles, self.S())

    def image_analyzer(self) -> ImageAnalyzer:
        return ImageAnalyzer(self.classifier, self.profiles, self.S())

    def train(self, log=print) -> dict:
        canon, y, groups = self.db.load_training_data()
        if len(y) == 0:
            raise ValueError("The dataset is empty. Capture or import labelled samples first.")
        meta = self.classifier.train(feature_matrix(canon), y, groups, log)
        self.db.add_training_run(meta)
        self.reload_profiles()
        return meta

    @staticmethod
    def new_group() -> str:
        return uuid.uuid4().hex[:10]

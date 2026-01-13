import hashlib
import json
import logging
import os
from enum import Enum
from typing import Any, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import PrivateAttr

from frigate.const import MODEL_CACHE_DIR
from frigate.plus import PlusApi

logger = logging.getLogger(__name__)


class PixelFormatEnum(str, Enum):
    rgb = "rgb"
    bgr = "bgr"
    yuv = "yuv"


class InputTensorEnum(str, Enum):
    nchw = "nchw"
    nhwc = "nhwc"
    hwnc = "hwnc"
    hwcn = "hwcn"


class InputDTypeEnum(str, Enum):
    float = "float"
    float_denorm = "float_denorm"  # non-normalized float
    int = "int"


class PoseModelTypeEnum(str, Enum):
    yolo_pose = "yolo_pose"
    mediapipe = "mediapipe"
    openpose = "openpose"
    movenet = "movenet"
    posenet = "posenet"
    mediapipe_task = "mediapipe_task"


class PoseModelConfig(BaseModel):
    path: Optional[str] = Field(None, title="Custom pose detection model path.")
    width: int = Field(default=320, title="Pose detection model input width.")
    height: int = Field(default=320, title="Pose detection model input height.")
    input_tensor: InputTensorEnum = Field(
        default=InputTensorEnum.nhwc, title="Model Input Tensor Shape"
    )
    input_pixel_format: PixelFormatEnum = Field(
        default=PixelFormatEnum.rgb, title="Model Input Pixel Color Format"
    )
    input_dtype: InputDTypeEnum = Field(
        default=InputDTypeEnum.int, title="Model Input D Type"
    )
    model_type: PoseModelTypeEnum = Field(
        default=PoseModelTypeEnum.yolo_pose, title="Pose Detection Model Type"
    )
    num_keypoints: int = Field(
        default=17, title="Number of keypoints the model detects"
    )
    keypoint_names: list[str] = Field(
        default=[
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        title="Names of keypoints in order",
    )
    confidence_threshold: float = Field(
        default=0.4, title="Minimum confidence threshold for pose detection"
    )
    keypoint_threshold: float = Field(
        default=0.3, title="Minimum confidence threshold for individual keypoints"
    )
    # fall_model_path: Optional[str] = Field(
    #     None, title="Custom model path for fall detection"
    # )

    _model_hash: str = PrivateAttr()

    @property
    def model_hash(self) -> str:
        return self._model_hash

    def __init__(self, **config):
        super().__init__(**config)

    def check_and_load_plus_model(
        self, plus_api: PlusApi, detector: str = None
    ) -> None:
        if not self.path or not self.path.startswith("plus://"):
            return

        # ensure that model cache dir exists
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        model_id = self.path[7:]
        self.path = os.path.join(MODEL_CACHE_DIR, model_id)
        model_info_path = f"{self.path}.json"

        # download the model if it doesn't exist
        if not os.path.isfile(self.path):
            download_url = plus_api.get_model_download_url(model_id)
            r = requests.get(download_url)
            with open(self.path, "wb") as f:
                f.write(r.content)

        # download the model info if it doesn't exist
        if not os.path.isfile(model_info_path):
            model_info = plus_api.get_model_info(model_id)
            with open(model_info_path, "w") as f:
                json.dump(model_info, f)
        else:
            with open(model_info_path, "r") as f:
                model_info: dict[str, Any] = json.load(f)

        if detector and detector not in model_info["supportedDetectors"]:
            raise ValueError(f"Model does not support detector type of {detector}")

        self.width = model_info["width"]
        self.height = model_info["height"]
        self.input_tensor = InputTensorEnum(model_info["inputShape"])
        self.input_pixel_format = PixelFormatEnum(model_info["pixelFormat"])
        self.model_type = PoseModelTypeEnum(model_info["type"])

        if model_info.get("inputDataType"):
            self.input_dtype = InputDTypeEnum(model_info["inputDataType"])

        if model_info.get("numKeypoints"):
            self.num_keypoints = model_info["numKeypoints"]

        if model_info.get("keypointNames"):
            self.keypoint_names = model_info["keypointNames"]

    def compute_model_hash(self) -> None:
        if not self.path or not os.path.exists(self.path):
            self._model_hash = hashlib.md5(b"unknown").hexdigest()
        else:
            with open(self.path, "rb") as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            self._model_hash = file_hash.hexdigest()

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class BasePoseDetectorConfig(BaseModel):
    # the type field must be defined in all subclasses
    type: str = Field(default="cpu", title="Pose Detector Type")
    model: Optional[PoseModelConfig] = Field(
        default=None, title="Pose detector specific model configuration."
    )
    model_path: Optional[str] = Field(
        default=None, title="Pose detector specific model path."
    )
    # Hardware accelerator to use for this detector (e.g. "edgetpu" or "cpu").
    accelerator: Optional[str] = Field(
        default=None, title="Hardware accelerator to use (edgetpu, gpu, etc.)"
    )
    # Device identifier for accelerator use (for example EdgeTPU usb device string).
    device: Optional[str] = Field(
        default=None, title="Accelerator device identifier (optional)"
    )
    # Number of threads to use for CPU/TFLite inference when applicable.
    num_threads: Optional[int] = Field(
        default=None, title="Number of interpreter threads to use (optional)"
    )
    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, protected_namespaces=()
    )

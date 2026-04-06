# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
from pytest import FixtureRequest
import json
import numpy as np
import re
import contextvars
from dataclasses import dataclass, is_dataclass, field
from dataclasses_json import dataclass_json
from typing import Union, List, Optional, Any, get_origin, get_args, Dict, Tuple
from forge.verify.config import VerifyConfig
from forge.verify.compare import determine_consistency_limits
from forge.config import CompilerConfig
from forge._C import DataFormat
from forge.tensor import Tensor, forge_dataformat_to_pytorch_dtype
from loguru import logger
from forge.module import ForgeModule
import forge
from torch import Tensor as TorchTensor


def get_device_arch() -> str:
    """
    Return the hardware architecture of the connected TT device.

    Returns:
        str: ``"wormhole"``, ``"blackhole"``, or ``""`` when no device is
        present or the arch cannot be determined.
    """
    try:
        from forge._C.runtime.experimental import TTSystem

        system = TTSystem.get_system()
        if not system.devices:
            return ""
        arch_name = str(system.devices[0].arch).lower()
        for family in ("wormhole", "blackhole"):
            if family in arch_name:
                return family
        return ""
    except Exception:
        return ""


def get_device_count() -> int:
    """
    Return the total number of TT chips visible to the host.

    Returns:
        int: Chip count, or ``0`` when the system cannot be queried.
    """
    try:
        from forge._C.runtime.experimental import TTSystem

        return len(TTSystem.get_system().chip_ids)
    except Exception:
        return 0


def get_device_type(arch: str, device_count: int) -> str:
    """
    Map a coarse architecture family and chip count to a product-label string.

    =============  ============  ===========
    arch           device_count  label
    =============  ============  ===========
    wormhole       1             ``n150``
    wormhole       2             ``n300``
    blackhole      1             ``p150``
    blackhole      2             ``p300``
    *any*          8             ``llmbox``
    *any*          32            ``galaxy``
    =============  ============  ===========

    Args:
        arch (str): Coarse arch string (``"wormhole"`` or ``"blackhole"``).
        device_count (int): Number of chips from the system descriptor.

    Returns:
        str: Product label, or ``"unknown"`` when no rule matches.
    """
    if device_count == 32:
        return "galaxy"
    if device_count == 8:
        return "llmbox"
    if arch == "wormhole":
        if device_count == 1:
            return "n150"
        if device_count == 2:
            return "n300"
    if arch == "blackhole":
        if device_count == 1:
            return "p150"
        if device_count == 2:
            return "p300"
    return "unknown"


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class BaseEnum(Enum):
    """Base Enum to handle short and full name attributes."""

    def __init__(self, short: str, full: str):
        self.short = short  # Short identifier (e.g., "hf" for Hugging Face)
        self.full = full  # Full descriptive name (e.g., "Hugging Face")


class Framework(BaseEnum):
    PYTORCH = ("pt", "PyTorch")
    TENSORFLOW = ("tf", "TensorFlow")
    ONNX = ("onnx", "ONNX")
    PADDLE = ("pd", "PaddlePaddle")
    JAX = ("jax", "JAX")


class Task(BaseEnum):
    # NLP Tasks
    NLP_SEQUENCE_CLASSIFICATION = ("nlp_text_cls", "NLP Sequence Classification")
    NLP_TOKEN_CLASSIFICATION = ("nlp_token_cls", "NLP Token Classification")
    NLP_TEXT_ENCODING = ("nlp_embed_gen", "NLP Text Encoding")
    NLP_QA = ("nlp_qa", "NLP Question Answering")
    NLP_CAUSAL_LM = ("nlp_causal_lm", "NLP Causal Language Modeling")
    NLP_MASKED_LM = ("nlp_masked_lm", "NLP Masked Language Modeling")
    NLP_TRANSLATION = ("nlp_translation", "NLP Translation")
    # Audio Tasks
    AUDIO_SPEECH_RECOGNITION = ("audio_asr", "Audio Speech Recognition")
    # CV Tasks
    CV_IMAGE_CLASSIFICATION = ("cv_image_cls", "CV Image Classification")
    CV_IMAGE_SEGMENTATION = ("cv_image_seg", "CV Image Segmentation")
    CV_OBJECT_DETECTION = ("cv_object_det", "CV Object Detection")
    CV_DEPTH_ESTIMATION = ("cv_depth_est", "CV Depth Estimation")
    CV_PLANNING_ORIENTED_DRIVING = ("cv_img_to_img", "CV Planning-Oriented Driving")
    CV_NOVEL_VIEW_SYNTHESIS = ("cv_img_to_img", "CV Novel View Synthesis")
    CV_IMAGE_ENCODING = ("cv_image_fe", "CV Image Encoding")
    CV_MASKED_IMAGE_MODELING = ("cv_mask_gen", "CV Masked Image Modeling")
    CV_POSE_ESTIMATION = ("cv_keypoint_det", "CV Pose Estimation")
    # MM Tasks
    MM_IMAGE_CAPTIONING = ("mm_image_capt", "MM Image Captioning")
    MM_IMAGE_TEXT_PAIRING = ("mm_visual_qa", "MM Image Text Pairing")
    MM_TEXT_TO_SPEECH = ("mm_tts", "MM Text to Speech")
    MM_MUSIC_GENERATION = ("mm_tts", "MM Music Generation")
    MM_TEXT_TO_VIDEO_GENERATION = ("mm_video_ttt", "MM Text-to-Video Generation")
    MM_CAUSAL_LM = ("mm_causal_lm", "MM Causal Language Modeling")
    # Other Tasks
    CONDITIONAL_GENERATION = ("conditional_generation", "Conditional Generation")
    ATOMIC_ML = ("atomic_ml", "Atomic Machine Learning")
    TIME_SERIES_FORECASTING = ("atomic_ml", "Time Series Forecasting")
    REALTIME_MAP_CONSTRUCTION = ("realtime_map_construction", "Realtime Map Construction")


class Source(BaseEnum):
    HUGGINGFACE = ("hf", "Hugging Face")
    TORCH_HUB = ("torch_hub", "Torch Hub")
    TIMM = ("timm", "TIMM")
    OSMR = ("osmr", "OSMR")
    TORCHVISION = ("torchvision", "Torchvision")
    GITHUB = ("github", "GitHub")
    PADDLE = ("paddlemodels", "Paddle Models")
    PADDLENLP = ("padlenlp", "PaddleNLP")
    KERAS = ("keras", "Keras")


class ModelArch(BaseEnum):
    RESNET = ("resnet", "ResNet")
    BERT = ("bert", "BERT")
    VIT = ("vit", "ViT")
    VITBASE = ("vit_base", "ViT Base")
    GPT = ("gpt", "GPT")
    STABLEDIFFUSION = ("stable_diffusion", "Stable Diffusion")
    LLAMA = ("llama", "LLaMA")
    OPENLLAMA = ("open_llama", "Open LLaMA")
    OFT = ("oft", "OFT")
    BIRNNCRF = ("BiRnnCrf", "BiRNN-CRF ")
    COGITO = ("cogito", "Cogito")
    GEMMA = ("gemma", "Gemma")
    LLAMA3 = ("llama3", "Llama 3")
    LLAMA3_2 = ("llama3_2", "Llama 3.2")
    MINILM = ("minilm", "MiniLM")
    MINISTRAL = ("ministral", "Ministral")
    MISTRAL = ("mistral", "Mistral")
    PHI1 = ("phi1", "Phi 1")
    PHI1_5 = ("phi_1.5", "Phi-1.5")
    PHI2 = ("phi2", "Phi 2")
    PHI3 = ("phi3", "Phi 3")
    PHI3_5 = ("phi3.5", "Phi-3.5")
    PHI35VISION = ("phi3_5_vision", "Phi-3.5 Vision")
    PHI4 = ("phi4", "Phi 4")
    DETR = ("detr", "DETR")
    DLA = ("dla", "DLA")
    EFFICIENTNET = ("efficientnet", "EfficientNet")
    EFFICIENTNETLITE = ("efficientnet_lite", "EfficientNet Lite")
    FALCON3 = ("falcon3", "Falcon 3")
    MOBILENETV1 = ("mobilenetv1", "MobileNetV1")
    MOBILENETV2 = ("mobilenetv2", "MobileNetV2")
    MOBILENETV2SSD = ("mobilenetv2_ssd", "MobileNetV2 SSD")
    MOBILENETV3 = ("mobilenetv3", "MobileNetV3")
    MOBILENETV3SSD = ("mobilenetv3_ssd", "MobileNetV3 SSD")
    SAM = ("sam", "SAM")
    SEGFORMER = ("segformer", "SegFormer")
    SWIN = ("swin", "Swin Transformer")
    UNET = ("unet", "UNet")
    UNETCARVANA = ("unet_carvana", "UNet Carvana")
    VOVNET = ("vovnet", "VoVNet")
    VILT = ("vilt", "ViLT")
    VOVNETV1 = ("vovnet_v1", "VoVNet V1")
    YOLOV3 = ("Yolo v3", "YOLOv3")
    YOLOV4 = ("Yolo v4", "YOLOv4")
    YOLOV5 = ("yolo_v5", "YOLOv5")
    YOLOV6 = ("yolo_v6", "YOLOv6")
    YOLOV8 = ("Yolov8", "YOLOv8")
    YOLOV9 = ("Yolov9", "YOLOv9")
    YOLOV10 = ("Yolov10", "YOLOv10")
    YOLOWORLD = ("yolo_world", "YOLO World")
    YOLOS = ("yolos", "YOLOS")
    YOLOX = ("yolox", "YOLOX")
    SPEECHT5 = ("speecht5", "SpeechT5")
    BLIP = ("blip", "BLIP")
    BLIPTEXT = ("blip_text", "BLIP Text")
    BLIPVISION = ("blip_vision", "BLIP Vision")
    CLIP = ("clip", "CLIP")
    CLIPTEXT = ("clip_text", "CLIP Text")
    CLIPVISION = ("clip_vision", "CLIP Vision")
    CHINESECLIP = ("chineseclip", "Chinese-CLIP")
    CHINESECLIPTEXT = ("chineseclip_text", "Chinese-CLIP Text")
    CHINESECLIPVISION = ("chineseclip_vision", "Chinese-CLIP Vision")
    PADDLEOCR = ("paddleocr", "Paddle OCR")
    ALBERT = ("albert", "ALBERT")
    BLOOM = ("bloom", "BLOOM")
    CODEGEN = ("codegen", "CodeGen")
    DISTILBERT = ("distilbert", "DistilBERT")
    DPR = ("dpr", "DPR")
    BART = ("bart", "BART")
    LLAVA = ("llava", "LLaVA")
    DEEPSEEK = ("deepseek", "DeepSeek")
    ERNIE = ("ernie", "ERNIE")
    FALCON = ("falcon", "Falcon")
    FUYU = ("fuyu", "Fuyu")
    GLINER = ("Gliner", "GLiNER")
    GPTNEO = ("gptneo", "GPT Neo")
    HIPPYNN = ("hippynn", "Hippynn")
    MAMBA = ("mamba", "Mamba")
    STEREO = ("stereo", "Stereo")
    WHISPER = ("whisper", "Whisper")
    NANOGPT = ("nanogpt", "NanoGPT")
    OPT = ("opt", "OPT")
    PERCEIVERIO = ("perceiverio", "Perceiver IO")
    QWENCODER = ("qwen_coder", "Qwen2.5-Coder")
    QWENV2 = ("qwen_v2", "Qwen2.5")
    QWEN15 = ("qwen1.5", "Qwen1.5")
    ROBERTA = ("roberta", "RoBERTa")
    SPEECHT5TTS = ("speecht5_tts", "SpeechT5 TTS")
    SQUEEZEBERT = ("squeezebert", "SqueezeBERT")
    T5 = ("t5", "T5")
    GLM = ("glm", "GLM")
    XGLM = ("xglm", "XGLM")
    NBEATS = ("nbeats", "N-BEATS")
    ALEXNET = ("alexnet", "AlexNet")
    AUTOENCODER = ("autoencoder", "Autoencoder")
    BEIT = ("beit", "BEiT")
    DEIT = ("deit", "DeiT")
    DENSENET = ("densenet", "DenseNet")
    FPN = ("fpn", "FPN")
    GHOSTNET = ("ghostnet", "GhostNet")
    HRNET = ("hrnet", "HRNet")
    GLPNKITTI = ("glpn_kitti", "GLPN KITTI")
    GOOGLENET = ("googlenet", "GoogLeNet")
    INCEPTION = ("inception", "Inception")
    MGP = ("mgp", "MGP-STR")
    MLPMIXER = ("mlp_mixer", "MLP-Mixer")
    MNIST = ("mnist", "MNIST")
    MONODEPTH2 = ("monodepth2", "Monodepth2")
    MONODLE = ("monodle", "MonoDLE")
    RCNN = ("rcnn", "R-CNN")
    REGNET = ("regnet", "RegNet")
    RESNEXT = ("resnext", "ResNeXt")
    RETINANET = ("retinanet", "RetinaNet")
    RMBG20 = ("rmbg_2_0", "RMBG-2.0")
    SSD300RESNET50 = ("ssd300_resnet50", "SSD300 ResNet50")
    SSD300VGG16 = ("ssd300_vgg16", "SSD300 VGG16")
    SSDLITE320MOBILENETV3 = ("ssdlite320_mobilenetv3", "SSD Lite MobileNetV3")
    VGG = ("vgg", "VGG")
    VGG19UNET = ("VGG19 UNet", "VGG19 UNet")
    WIDERESNET = ("wideresnet", "Wide ResNet")
    XCEPTION = ("xception", "Xception")
    MAPTR = ("maptr", "MapTR")
    PHI3_5_MOE = ("phi3.5_moe", "Phi-3.5-MoE")
    UNIAD = ("uniad", "UniAD")
    MINIMAX = ("minimax", "MiniMax")
    MPLUGOWL = ("mplug_owl", "mPLUG-Owl")
    CENTERNET = ("centernet", "Centernet")
    SURYAOCR = ("surya_ocr", "Surya_OCR")
    TRANSFUSER = ("transfuser", "Transfuser")
    BEVDEPTH = ("bevdepth", "Bevdepth")
    POINTPILLARS = ("pointpillars", "Pointpillars")
    BEVFORMER = ("bevformer", "Bevformer")
    FLUX = ("flux", "Flux")
    PIXTRAL = ("pixtral", "Pixtral")
    SOLAR = ("solar", "Solar")
    VADV2 = ("vadv2", "Vadv2")
    QWENV3 = ("qwen_v3", "Qwen_v3")
    GAUSSIAN_SPLATTING = ("gaussian_splatting", "Gaussian Splatting")
    DETR3D = ("detr_3D", "Detr-3D")
    MOCHIV1 = ("mochi-1", "Mochi-1")
    TRANKIT = ("trankit", "Trankit")
    MPLUGOWL2 = ("mplug_owl2", "mPLUG-Owl2")
    LLAMA4 = ("llama4", "Llama-4")
    MIXTRAL = ("mixtral", "Mixtral")


def build_module_name(
    framework: Framework,
    model: str,
    task: Task,
    source: Source,
    variant: str = "base",
    suffix: str | None = None,
) -> str:
    module_name = f"{framework}_{model}_{variant}_{task}_{source}"
    if suffix is not None:
        module_name += f"_{suffix}"
    module_name = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
    module_name = re.sub(r"_+", "_", module_name)
    module_name = module_name.lower()
    return module_name


class ExecutionRunMode(BaseEnum):
    INFERENCE = ("inference", "INFERENCE")
    TRAINING = ("training", "TRAINING")

    @classmethod
    def from_training_param(cls, training: bool):
        return cls.TRAINING if training else cls.INFERENCE


class ExecutionPass(Enum):
    FORWARD = ("FORWARD", "forward")
    BACKWARD = ("BACKWARD", "backward")

    @classmethod
    def to_str(cls, value):
        return value.name

    @classmethod
    def from_str(cls, value):
        return cls[value.upper()]


class ExecutionStage(Enum):
    FAILED_BEFORE_FORGE_COMPILATION_INITIATION = auto()
    FAILED_TVM_RELAY_IRMODULE_GENERATION = auto()
    FAILED_TVM_RELAY_IO_FLATTENING = auto()
    FAILED_TVM_RELAY_IR_TRANSFORMATION = auto()
    FAILED_TVM_PATTERN_CALLBACKS = auto()
    FAILED_TVM_GRAPH_PARTITIONING = auto()
    FAILED_FORGE_MODULE_GENERATION = auto()
    FAILED_FORGE_INITIAL_GRAPH_PASS = auto()
    FAILED_FORGE_POST_INITIAL_GRAPH_PASS = auto()
    FAILED_FORGE_CONSTEVAL = auto()
    FAILED_FORGE_OPTIMIZATION_GRAPH_PASS = auto()
    FAILED_FORGE_POST_OPTIMIZATION_DECOMP = auto()
    FAILED_FORGE_AUTOGRAD_PASS = auto()
    FAILED_FORGE_POST_AUTOGRAD_DECOMP = auto()
    FAILED_FORGE_PRE_LOWERING = auto()
    FAILED_FORGE_GRAPH_SPLIT = auto()
    FAILED_FORGE_MLIR_COMPILATION = auto()
    FAILED_TTNN_BINARY_EXECUTION = auto()
    FAILED_VERIFICATION = auto()
    PASSED = auto()

    @classmethod
    def to_str(cls, value):
        return value.name

    @classmethod
    def from_str(cls, value):
        return cls[value.upper()]


class ExecutionDepth(Enum):
    CI_FAILURE = auto()
    FAILED_FE_COMPILATION = auto()
    FAILED_TTMLIR_COMPILATION = auto()
    FAILED_RUNTIME = auto()
    INCORRECT_RESULT = auto()
    PASSED = auto()

    @classmethod
    def to_str(cls, value):
        return value.name

    @classmethod
    def from_str(cls, value):
        return cls[value.upper()]

    # fmt: off
    @staticmethod
    def from_exec_stage(exec_stage: ExecutionStage):
        match exec_stage:
            case ExecutionStage.FAILED_BEFORE_FORGE_COMPILATION_INITIATION:
                return ExecutionDepth.CI_FAILURE
            case ExecutionStage.FAILED_TVM_RELAY_IRMODULE_GENERATION | ExecutionStage.FAILED_TVM_RELAY_IO_FLATTENING | ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION | ExecutionStage.FAILED_TVM_PATTERN_CALLBACKS \
                | ExecutionStage.FAILED_TVM_GRAPH_PARTITIONING | ExecutionStage.FAILED_FORGE_MODULE_GENERATION | ExecutionStage.FAILED_FORGE_INITIAL_GRAPH_PASS | ExecutionStage.FAILED_FORGE_POST_INITIAL_GRAPH_PASS \
                | ExecutionStage.FAILED_FORGE_CONSTEVAL | ExecutionStage.FAILED_FORGE_OPTIMIZATION_GRAPH_PASS | ExecutionStage.FAILED_FORGE_POST_OPTIMIZATION_DECOMP | ExecutionStage.FAILED_FORGE_AUTOGRAD_PASS \
                | ExecutionStage.FAILED_FORGE_POST_AUTOGRAD_DECOMP | ExecutionStage.FAILED_FORGE_PRE_LOWERING | ExecutionStage.FAILED_FORGE_GRAPH_SPLIT:
                return ExecutionDepth.FAILED_FE_COMPILATION
            case ExecutionStage.FAILED_FORGE_MLIR_COMPILATION:
                return ExecutionDepth.FAILED_TTMLIR_COMPILATION
            case ExecutionStage.FAILED_TTNN_BINARY_EXECUTION:
                return ExecutionDepth.FAILED_RUNTIME
            case ExecutionStage.FAILED_VERIFICATION:
                return ExecutionDepth.INCORRECT_RESULT
            case ExecutionStage.PASSED:
                return ExecutionDepth.PASSED
            case _:
                raise ValueError("Invalid ExecutionStage passed.")
    # fmt: on


@dataclass_json
@dataclass
class TensorDesc:
    shape: List[int]
    data_type: str = ""
    buffer_type: str = ""
    layout: str = ""
    grid_shape: Optional[List[int]] = None


class FlatbufferDetailsExtractor:
    """
    A utility class to parse and extract comprehensive details from a generated flatbuffer binary JSON.

    Args:
        binary_json (Dict[str, Any]): The flatbuffer binary JSON containing program details.
    """

    def __init__(self, binary_json):
        self.binary = binary_json

    def extract_tensor_details(self, inputs_or_outputs):
        """
        Extracts tensor descriptions from a list of input/output entries.

        Parameters:
            inputs_or_outputs (list): A list of dictionaries, each containing a "desc" key
                                      with tensor description details.

        Returns:
            list: A list of TensorDesc objects.
        """
        tensor_desc_list = []
        for input_or_output in inputs_or_outputs:
            desc = input_or_output["desc"]
            if (
                "shape" in desc
                and "layout" in desc
                and "memory_desc" in desc["layout"]
                and "data_type" in desc["layout"]["memory_desc"]
            ):
                tensor_desc = TensorDesc(shape=desc["shape"], data_type=desc["layout"]["memory_desc"]["data_type"])

                try:
                    tensor_desc.buffer_type = desc["layout"]["memory_desc"]["memory_space"]
                except KeyError:
                    if "memory_config" in desc["layout"]["memory_desc"]:
                        # If the tensor is on device, the descriptor will have a "memory_config" field.
                        tensor_desc.buffer_type = desc["layout"]["memory_desc"]["memory_config"]["buffer_type"]
                    else:
                        # If the tensor is on host, the descriptor will have a "storage_type" field.
                        tensor_desc.buffer_type = desc["layout"]["memory_desc"]["storage_type"]

                try:
                    tensor_desc.layout = desc["layout"]["memory_desc"]["memory_layout"]
                except KeyError:
                    if "memory_config" in desc["layout"]["memory_desc"]:
                        # If the tensor is on device, use the tensor_memory_layout from memory_config.
                        tensor_desc.layout = desc["layout"]["memory_desc"]["memory_config"]["tensor_memory_layout"]
                    else:
                        # If the tensor is on host, no tensor_memory_layout is available.
                        tensor_desc.layout = ""

                try:
                    grid_shape = desc["layout"]["core_range_set"][0]["size"]
                    tensor_desc.grid_shape = [grid_shape["x"], grid_shape["y"]]
                except KeyError:
                    pass

                tensor_desc_list.append(tensor_desc)
        return tensor_desc_list

    def extract_program_io_details(self, program_filter: Optional[List[str]] = None):
        """
        Extracts detailed input and output configurations for each program from the flatbuffer binary JSON.

        Args:
            program_filter (Optional[List[str]]): A list of program names to filter the extraction process.
                Only programs whose names appear in this list will have their input/output details extracted.
                If None, details for all programs will be extracted.
        Returns:
            tuple: A tuple (program_inputs, program_outputs) where:
                - program_inputs (Dict[str, List[TensorDesc]]): Maps program names to detailed input configurations.
                - program_outputs (Dict[str, List[TensorDesc]]): Maps program names to detailed output configurations.
            Returns (None, None) if the binary JSON does not contain a "programs" key.
        """
        if "programs" not in self.binary:
            return None, None

        program_inputs = {}
        program_outputs = {}

        for program in self.binary["programs"]:
            program_name = program["name"]
            if program_filter is not None and program_name not in program_filter:
                continue
            inputs = self.extract_tensor_details(program["inputs"])
            outputs = self.extract_tensor_details(program["outputs"])
            if len(inputs) > 0:
                program_inputs[program_name] = inputs
            if len(outputs) > 0:
                program_outputs[program_name] = outputs

        return program_inputs, program_outputs


@dataclass_json
@dataclass
class Operand:
    node_type: str = ""
    shape: Optional[Tuple[int, ...]] = None
    dataformat: str = ""
    torch_dtype: str = ""


@dataclass_json
@dataclass
class OpInfo:
    forge_op_name: str = ""
    args: Dict[str, Any] = field(default_factory=lambda: dict())
    operands: Optional[List[Operand]] = None
    model_names: Optional[List[str]] = None


@dataclass_json
@dataclass
class Config:
    compiler: Dict[str, Any] = field(default_factory=lambda: dict())
    verify: Dict[str, Any] = field(default_factory=lambda: dict())


# Model group property that is part of a model info. With current reports we are using tags: 'group' and 'tags.group'.
# If we want to add group there too, we would need to change how reporters read this info.
class ModelGroup(StrEnum):
    GENERALITY = "generality"
    RED = "red"


# Model priority property that is part of a model info. With current reports we are using tag: 'priority'.
# If we want to add priority in model's info too, we would need to change how reporters read this info.
class ModelPriority(StrEnum):
    P1 = "P1"
    P2 = "P2"


class Parallelism(StrEnum):
    SINGLE_DEVICE = "single_device"
    DATA_PARALLEL = "data_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


@dataclass_json
@dataclass
class ModelInfo:
    framework: str = ""
    model_arch: str = ""
    variant_name: str = ""
    task: str = ""
    source: str = ""


@dataclass_json
@dataclass
class SweepsTags:
    operator: str = ""
    input_source: str = ""
    input_shape: str = ""
    dev_data_format: str = ""
    math_fidelity: str = ""
    kwargs: str = ""
    expected_failing_reason: str = ""
    expected_failing_reason_desc: str = ""
    expected_component: str = ""
    detected_failing_reason: str = ""
    detected_failing_reason_desc: str = ""
    detected_component: str = ""


@dataclass_json
@dataclass
class Tags:
    model_name: Optional[str] = None
    bringup_status: str = ""
    execution_stage: str = ""
    run_mode: str = ""
    execution_pass: str = ""
    pcc: Optional[float] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    op_info: Optional[OpInfo] = None
    inputs: Optional[List[TensorDesc]] = None
    outputs: Optional[List[TensorDesc]] = None
    model_info: Optional[ModelInfo] = None
    sweeps: Optional[SweepsTags] = None
    failure_category: str = ""
    refined_error_message: str = ""
    group: Optional[str] = None
    parallelism: Optional[Parallelism] = Parallelism.SINGLE_DEVICE.value
    emitc_status: Optional[bool] = None
    arch: Optional[str] = None


@dataclass_json
@dataclass
class ForgePropertyStore:
    owner: str = "tt-forge-onnx"
    group: Optional[str] = None
    priority: Optional[str] = None
    tags: Optional[Tags] = None
    config: Optional[Config] = None


class ForgePropertyHandler:
    """
    Handles storage and retrieval of properties in a nested ForgePropertyStore.

    This class provides methods to add, update, retrieve, and clean properties stored in a
    ForgePropertyStore. It supports nested attributes using dot-separated keys and includes
    several utility methods for recording specific property values (such as group, model name,
    and configuration data).

    Default initial execution stage (FAILED_BEFORE_FORGE_COMPILATION_INITIATION) is set.

    Attributes:
        store (ForgePropertyStore): The underlying store containing the property data.
        should be recorded.
    """

    def __init__(self, store: Optional[ForgePropertyStore] = None):
        self.store = store if store is not None else ForgePropertyStore()
        self.record_execution(ExecutionStage.FAILED_BEFORE_FORGE_COMPILATION_INITIATION)

    def add(self, key: str, value: Any):
        """
        Adds or updates a property in the store using a dot-separated key.

        The dot-separated key indicates the nested structure. For example, a key "tags.model_name"
        means that the 'model_name' attribute is under the 'tags' attribute of the store. If any
        intermediate attribute is missing or is None, it is created using a default instance.

        Args:
            key (str): Dot-separated string representing the property path.
            value (Any): The value to be assigned at the specified property path.

        Raises:
            ValueError: If the provided key is empty.
        """
        if not key:
            raise ValueError("Key cannot be empty.")
        keys = key.split(".")
        obj = self.store
        # Traverse the nested structure, creating missing attributes as needed.
        for k in keys[:-1]:
            if not hasattr(obj, k) or getattr(obj, k) is None:
                default_instance = self._create_default_instance(obj, k)
                setattr(obj, k, default_instance)
            obj = getattr(obj, k)
        # Set the final attribute to the provided value.
        setattr(obj, keys[-1], value)

    def __call__(self, key: str, value: Any):
        """
        Allows the handler to be called like a function to add or update a property.

        Example:
            handler("tags.op_name", "relu")

        Args:
            key (str): Dot-separated key path for the property.
            value (Any): The value to be set.
        """
        self.add(key, value)

    def get(self, key: str):
        """
        Retrieves a property value using a dot-separated key.

        The function traverses the nested attributes as specified by the key and returns the value.
        If any attribute in the chain is missing or is None, the function returns None.

        Args:
            key (str): Dot-separated string representing the property path.

        Returns:
            Any: The value of the property, or None if any attribute in the path is missing.

        Raises:
            ValueError: If the provided key is empty.
        """
        if not key:
            raise ValueError("Key cannot be empty.")
        keys = key.split(".")
        obj = self.store
        # Walk through the nested structure.
        for k in keys:
            if not hasattr(obj, k):
                return None
            obj = getattr(obj, k)
            if obj is None:
                return None
        return obj

    def _create_default_instance(self, parent: Any, attr: str):
        """
        Creates a default instance for a missing attribute based on the parent's dataclass metadata.

        For Optional fields that have a dataclass as their inner type, this method will instantiate
        the inner type. If a suitable dataclass cannot be inferred, it falls back to returning an empty dict.

        Args:
            parent (Any): The parent object, typically a dataclass instance.
            attr (str): The name of the attribute for which a default instance is needed.

        Returns:
            Any: A new instance of the attribute's type, or an empty dictionary if instantiation fails.
        """
        if hasattr(parent, "__dataclass_fields__"):
            field_info = parent.__dataclass_fields__.get(attr)
            if field_info is not None:
                field_type = field_info.type
                # Handle Optional[T] by extracting T from Union[T, None].
                origin = get_origin(field_type)
                if origin is Union:
                    args = get_args(field_type)
                    non_none_types = [arg for arg in args if arg is not type(None)]
                    if len(non_none_types) == 1:
                        field_type = non_none_types[0]
                if is_dataclass(field_type):
                    return field_type()
        # Fallback: return an empty dictionary if no dataclass instance can be created.
        return {}

    def record_execution_depth(self, execution_depth: ExecutionDepth):
        """
        Records the execution depth (as bringup_status) in the tags.

        Args:
            execution_depth (ExecutionDepth): The execution depth value.
        """
        self.add("tags.bringup_status", ExecutionDepth.to_str(execution_depth))

    def record_execution_stage(self, execution_stage: ExecutionStage):
        """
        Records the execution stage in the tags.

        Args:
            execution_stage (ExecutionStage): The execution stage value.
        """
        self.add("tags.execution_stage", ExecutionStage.to_str(execution_stage))

    def record_execution_run_mode(self, execution_run_mode: ExecutionRunMode):
        """
        Records the execution run mode (as run_mode)in the tags.

        Args:
            execution_run_mode (ExecutionRunMode): The execution run mode value.
        """
        self.add("tags.run_mode", execution_run_mode.short)

    def record_execution_pass(self, execution_pass: ExecutionPass):

        """
        Records the execution pass (as execution_pass) in the tags.

        Args:
            execution_pass (ExecutionPass): The execution pass value.
        """
        self.add("tags.execution_pass", ExecutionPass.to_str(execution_pass))

    def record_execution(self, execution_stage: ExecutionStage):
        """
        Records the execution depth and stage in the tags.

        Args:
            execution_stage (ExecutionStage): The execution stage value.
        """
        self.record_execution_stage(execution_stage)
        self.record_execution_depth(ExecutionDepth.from_exec_stage(execution_stage))

    def record_emitc_status(self, is_success: bool):
        self.add("tags.emitc_status", is_success)

    def record_arch(self, arch: str):
        """
        Records the hardware architecture label (e.g. "n150", "n300", "p150") in the tags.

        Args:
            arch (str): Product-label string for the connected TT device.
        """
        self.add("tags.arch", arch)

    def extract_node_type(self, operand):
        if isinstance(operand, forge.Parameter):
            return "Parameter"
        elif operand.is_constant():
            return "Constant"
        else:
            return "Activation"

    def to_dict(self):
        """
        Converts the entire property store to a dictionary.

        Returns:
            dict: The property store represented as a dictionary.
        """
        return self.store.to_dict()

    def items(self):
        """
        Returns an iterator over the property store items.

        Returns:
            ItemsIterator: An iterator over the key-value pairs in the property store.
        """
        return self.to_dict().items()

    def clean_store(self) -> dict:
        """
        Returns a cleaned dictionary version of the underlying store.

        A value is considered empty if it is None or an empty container (empty string, list,
        tuple, dict, or set). All keys with empty values are removed, except for keys named
        "compiler" or "verify". For those keys, the entire dict value is preserved as is,
        even if it contains empty members.

        Returns:
            dict: A cleaned version of the property store with empty values removed.
        """

        def is_empty(value):
            # Returns True if the value is None or an empty container.
            if value is None:
                return True
            if isinstance(value, (str, list, tuple, dict, set)) and len(value) == 0:
                return True
            return False

        def recursive_clean(data):
            # Recursively cleans a dictionary by removing keys with empty values.
            if not isinstance(data, dict):
                return data
            cleaned = {}
            for key, value in data.items():
                # Preserve keys "compiler" and "verify" as is.
                if key in ("compiler", "verify"):
                    cleaned[key] = value
                    continue

                if isinstance(value, dict):
                    cleaned_value = recursive_clean(value)
                else:
                    cleaned_value = value

                if not is_empty(cleaned_value):
                    cleaned[key] = cleaned_value
            return cleaned

        store_dict = self.store.to_dict()
        return recursive_clean(store_dict)

    def record_error(self, request: FixtureRequest):
        """
        Records refined error message and failure category if they exist.

        Parameters:
            request: Fixture request for current test.
        """
        # Retrieve any refined error message that might have been set during the test execution
        refined_error_message = getattr(request.node, "refined_error_message", None)
        if refined_error_message is None:
            return

        self.add("tags.refined_error_message", refined_error_message)

        # Add failure_category if it exist.
        failure_category = getattr(request.node, "failure_category", None)
        if failure_category is None:
            return

        self.add("tags.failure_category", failure_category)

    def record_all_properties(self, record_property):
        """
        Stores the cleaned properties using a provided recording function.

        Args:
            record_property (Callable): A function that accepts two arguments (property_name and
                property_value) to record the value.
        """
        cleaned_property_store = self.clean_store()
        for property_name, property_value in cleaned_property_store.items():
            record_property(property_name, property_value)


# Context var used for storing test properties without passing forge_property_handler as a parameter to all functions.
forge_property_handler_var = contextvars.ContextVar("forge_property_handler_var", default=None)

# Next section contains global recording functions. They all use forge_property_handler context variable to
# record various properties.


def record_execution_run_mode(execution_run_mode: ExecutionRunMode):
    """
    Records the execution run mode in the tags.

    Args:
        execution_run_mode (ExecutionRunMode): The execution run mode value.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.record_execution_run_mode(execution_run_mode)


def record_execution_pass(execution_pass: ExecutionPass):
    """
    Records the execution pass in the tags.

    Args:
        execution_pass (ExecutionPass): The execution pass value.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.record_execution_pass(execution_pass)


def record_execution(execution_stage: ExecutionStage):
    """
    Records the execution depth and stage in the tags.

    Args:
        execution_stage (ExecutionStage): The execution stage value.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.record_execution(execution_stage)


def record_emitc_status(is_success: bool):
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.record_emitc_status(is_success)


def record_arch(arch: str):
    """
    Records the hardware architecture label (e.g. "n150", "n300", "p150") in the tags.

    Args:
        arch (str): Product-label string for the connected TT device.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.record_arch(arch)


def record_compiler_config(compiler_config: CompilerConfig):
    """
    Records the compiler configuration under config.compiler.

    Args:
        compiler_config (CompilerConfig): The compiler configuration object.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.add("config.compiler", compiler_config.to_dict())


def record_verify_config(verify_config: VerifyConfig):
    """
    Records the verify configuration under config.verify.

    Converts the verify configuration to a dictionary, and ensures that the value
    for the 'value_checker' key is also represented as a dictionary.

    Args:
        verify_config (VerifyConfig): The verify configuration object.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    verify_config = verify_config.to_dict()
    verify_config["value_checker"] = verify_config["value_checker"].__dict__

    fph.add("config.verify", verify_config)


def record_flatbuffer_details(binary_json_str: str):
    """
    Records details (forward program inputs/outputs tensor description) from a flatbuffer binary JSON string.

    This method convert provided JSON string into a dictionary, and uses the
    FlatbufferDetailsExtractor to extract details and record it.

    Args:
        binary_json_str (str): The JSON string representation of the flatbuffer binary.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    if fph.get("tags.model_name"):
        # For model tests, we don't want to record the flatbuffer details, since this
        # results in a lot of data being recorded.
        return

    binary_json_str = re.sub(r":\s*-inf\s*([,}])", r': "-inf"\1', binary_json_str)
    binary_json_str = re.sub(r":\s*inf\s*([,}])", r': "inf"\1', binary_json_str)
    binary_json = json.loads(binary_json_str)

    flatbuffer_details_extractor = FlatbufferDetailsExtractor(binary_json)
    inputs, outputs = flatbuffer_details_extractor.extract_program_io_details(program_filter=["forward"])
    if inputs is not None and outputs is not None:
        if len(inputs) != len(outputs):
            logger.error(
                f"Mismatch in program count: inputs have {len(inputs)} programs, while outputs have {len(outputs)} programs."
            )
        if sorted(inputs.keys()) != sorted(outputs.keys()):
            logger.error(
                f"Mismatch in program names: inputs contain {sorted(inputs.keys())}, while outputs contain {sorted(outputs.keys())}."
            )

        fph.add("tags.inputs", inputs["forward"])
        fph.add("tags.outputs", outputs["forward"])


def record_model_properties(
    framework: Framework,
    model: ModelArch,
    task: Task,
    source: Source,
    variant: str = "base",
    suffix: str | None = None,
    group: ModelGroup = ModelGroup.GENERALITY,
    priority: ModelPriority = ModelPriority.P2,
) -> str:
    """
    Records model properties and generates a module name and stores it.

    Args:
        framework: The framework used (e.g., pt,tf, etc.)
        model: The model name (e.g., bert)
        variant: The model variant (e.g., bert-base-uncased)
        task: The task type (e.g., qa,mlm, etc.)
        source: The model source (e.g., hf,torchhub etc.)
        suffix: Optional suffix to append to the module name
        group: The model group
        priority: The model priority

    Returns:
        The generated module name
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    if not isinstance(variant, str):
        variant = str(variant)

    # Record individual properties
    fph.add("tags.model_info.framework", framework.short)
    fph.add("tags.model_info.model_arch", model.short)
    fph.add("tags.model_info.variant_name", variant)
    fph.add("tags.model_info.task", task.short)
    fph.add("tags.model_info.source", source.short)

    # This should also be tagged with: tags.model_info.<priority/group>, but it requires changes in reporter too.
    # Leaving it as it is for now.
    # self.add("tags.model_info.priority", priority.value)
    # self.add("tags.model_info.group", group.value)

    fph.add("group", group.value)
    fph.add("tags.group", group.value)
    fph.add("priority", priority.value)

    # Build and return the module name
    module_name = build_module_name(
        framework=framework.short,
        model=model.short,
        variant=variant,
        task=task.short,
        source=source.short,
        suffix=suffix,
    )

    # Record model_name
    fph.add("tags.model_name", module_name)
    return module_name


def sanitize_json_float(value: Optional[float]) -> Optional[float]:
    """
    Sanitize float values for JSON serialization.

    Parameters:
        value: The float value to sanitize.

    Returns:
        The sanitized float value, or None if the value is NaN or Inf.
    """
    if value is None or np.isnan(value) or np.isinf(value):
        return None
    return value


def record_consistency_limits(
    framework_outputs: Union[Tuple[TorchTensor, ...], List[TorchTensor]], compiled_outputs: List[TorchTensor]
):
    """
    Records consistency limits (PCC, ATOL, and RTOL).

    Parameters:
        framework_outputs: Tuple or list of torch.Tensor representing the expected (golden) outputs.
        compiled_outputs: List of torch.Tensor representing the computed outputs.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    pcc, atol, rtol = determine_consistency_limits(
        framework_outputs=framework_outputs, compiled_outputs=compiled_outputs
    )
    pcc = sanitize_json_float(pcc)
    atol = sanitize_json_float(atol)
    rtol = sanitize_json_float(rtol)
    if pcc is not None:
        fph.add("tags.pcc", pcc)
    if atol is not None:
        fph.add("tags.atol", atol)
    if rtol is not None:
        fph.add("tags.rtol", rtol)


def record_sweeps_test_tags(
    operator: str,
    input_source: str = None,
    input_shape: str = None,
    dev_data_format: str = None,
    math_fidelity: str = None,
    kwargs: str = None,
):
    """
    Records the operator and additional tags in the sweeps tags.
    Args:
        operator (str): The operator name.
        input_source (str): The source of the input data.
        dev_data_format (str): The data format used in the device.
        math_fidelity (str): The math fidelity level.
        input_shape (str): The shape of the input data.
        kwargs (str): Additional keyword arguments to be recorded.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    if operator is not None:
        fph.add("tags.sweeps.operator", operator)
    if input_source is not None:
        fph.add("tags.sweeps.input_source", input_source)
    if input_shape is not None:
        fph.add("tags.sweeps.input_shape", input_shape)
    if dev_data_format is not None:
        fph.add("tags.sweeps.dev_data_format", dev_data_format)
    if math_fidelity is not None:
        fph.add("tags.sweeps.math_fidelity", math_fidelity)
    if kwargs is not None:
        fph.add("tags.sweeps.kwargs", kwargs)


def record_sweeps_expected_failing_reason(
    expected_failing_reason: str = "",
    expected_failing_reason_desc: str = "",
    expected_component: str = "",
):
    """
    Records the operator and failing reason in the sweeps tags.

    Args:
        operator (str): The operator name.
        expected_failing_reason (str): The expected failing reason.
        expected_failing_reason_desc (str): The description of the expected failing reason.
        expected_component (str): The expected component.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    if expected_failing_reason is not None:
        fph.add("tags.sweeps.expected_failing_reason", expected_failing_reason)
    if expected_failing_reason_desc is not None:
        fph.add("tags.sweeps.expected_failing_reason_desc", expected_failing_reason_desc)
    if expected_component is not None:
        fph.add("tags.sweeps.expected_component", expected_component)


def record_sweeps_detected_failing_reason(
    detected_failing_reason: str = "",
    detected_failing_reason_desc: str = "",
    detected_component: str = "",
):
    """
    Records detected failing reason and its description in the sweeps tags.

    Args:
        detected_failing_reason (str): The detected failing reason.
        detected_failing_reason_desc (str): The description of the detected failing reason.
        detected_component (str): The detected component.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    if detected_failing_reason is not None:
        fph.add("tags.sweeps.detected_failing_reason", detected_failing_reason)
    if detected_failing_reason_desc is not None:
        fph.add("tags.sweeps.detected_failing_reason_desc", detected_failing_reason_desc)
    if detected_component is not None:
        fph.add("tags.sweeps.detected_component", detected_component)


def record_forge_op_name(forge_op_name: str):
    """
    Records the Forge op name in the op information tags if single op details recording is enabled.

    Args:
        forge_op_name (str): The Forge operation name.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.add("tags.op_info.forge_op_name", forge_op_name)


def record_forge_op_args(op_args: Dict[str, Any]):
    """
    Records the arguments for the Forge operation in the op information tags if single op details recording is enabled.

    Args:
        op_args (Dict[str, Any]): A dictionary of operation arguments.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.add("tags.op_info.args", op_args)


def record_single_op_operands_info(forge_module: ForgeModule, inputs: List[Tensor]):
    """
    Records details about operation operands in the op information tags if single op details recording is enabled.

    For each operand, the function records the node type, shape, dataformat,
    and the corresponding PyTorch data type.

    forge_module (List[str]): ForgeModule to extract the operands details
    inputs (List[str]): List of forge tensor inputs for the module
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    assert isinstance(
        forge_module, ForgeModule
    ), f"Operands details can be extracted only from the ForgeModule but you have provided {forge_module}"
    output = forge_module(*inputs)
    assert isinstance(output, Tensor), "ForgeModule should have only one output tensor"
    if output.src_op is not None:
        assert all(
            True if isinstance(operand, forge.Parameter) or operand.src_op is None else False
            for operand in output.src_op.operands
        ), "ForgeModule should contains single forge op"
        operands_list = []
        for operand in output.src_op.operands:
            node_type = fph.extract_node_type(operand)
            shape = tuple(operand.shape.get_pytorch_shape())
            dataformat = DataFormat.to_json(operand.data_format)
            torch_dtype = str(operand.pt_data_format)
            operands_list.append(
                Operand(
                    node_type=node_type,
                    shape=shape,
                    dataformat=dataformat,
                    torch_dtype=str(torch_dtype),
                )
            )
        fph.add("tags.op_info.operands", operands_list)


def record_op_model_names(model_names: List[str]):
    """
    Records the model names associated with the operation in the op information tags if single op details recording is enabled.

    Args:
        model_names (List[str]): A list of model names.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.add("tags.op_info.model_names", model_names)


def record_parallelism(parallelism: Parallelism):
    """
    Records the paralleism property inside the tags.
    """
    fph = forge_property_handler_var.get()
    if fph is None:
        return

    fph.add("tags.parallelism", parallelism.value)


def get_model_group():
    fph = forge_property_handler_var.get()
    if fph is None:
        return None

    return fph.get("tags.group")

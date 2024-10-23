import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (Profile, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


class Identify:
    def __init__(self):
        self.cap = cv2.VideoCapture()
        # model dependency
        self.weights = ROOT / 'A_pt/best.pt'  # Set the weight model for recognition
        self.conf_thres = 0.25  # Identification confidence threshold
        self.iou_thres = 0.25  # Crossover ratio IoU threshold
        self.img_size = 640  # Size of the network input picture when predicting, default value is [640]
        self.classes = None  # Specifies the target category to detect. The default is None, indicating that all categories are detected
        self.max_det = 1000  # Maximum number of boxes detected per image, default is 1000.
        self.agnostic_nms = False  # Whether to use class-independent non-maximum suppression, which defaults to False
        self.line_thickness = 3  # Check the line width of the box. The default value is 3
        self.device = ''  # The device used can be the ID of the CUDA device (e.g. 0, 0,1,2,3) or 'cpu', which defaults to '0'.
        self.augment = False  # Whether to use data enhancement for inference
        self.visualize = False  # No Visualize the feature map in the model. The default is False
        self.dnn = False  # Whether to use OpenCV DNN for ONNX inference
        self.half = False  # Whether to use FP16 semi-precision for inference
        self.class_nums = 0  # The number of types identified
        """====================================Loading model========================================"""
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data='', fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.class_nums = len(self.names)  # Gets the number of identifiers

    # YOLOv5 detects core functions
    # Input: Input image, camera open identifier
    # Output: input image, output image, detection content list
    def show_frame(self, image, cap_flag):
        if cap_flag:
            flag, image = self.cap.read()
        if image is not None:
            img = image.copy()
            show_img = img
            labels = []
            with torch.no_grad():
                # 图像转换
                img = letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]  # padded resize
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)  # contiguous
                dt = (Profile(), Profile(), Profile())
                with dt[0]:
                    img = torch.from_numpy(img).to(self.model.device)
                    img = img.half() if self.half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if len(img.shape) == 3:
                        img = img[None]  # expand for batch dim
                # Inference
                with dt[1]:
                    pred = self.model(img, augment=self.augment, visualize=self.visualize)
                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                               self.classes, self.agnostic_nms, max_det=self.max_det)
                    # Process detectionss
                annotator = Annotator(show_img, line_width=self.line_thickness, example=str(self.names))
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], show_img.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = f'{self.names[c]} {conf:.2f}'
                            labels.append(self.names[c])  # 获取识别类别
                            annotator.box_label(xyxy, label, color=colors(c, True))
                show_img = annotator.result()
            return image, show_img, labels
        else:
            return image, None, None

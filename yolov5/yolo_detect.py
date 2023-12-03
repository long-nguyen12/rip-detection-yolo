# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import requests
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import datetime

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from constants import Constants
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from yolov5.utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams,
)
from yolov5.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from yolov5.utils.torch_utils import select_device, smart_inference_mode
import numpy as np
from app.utils.env_service import env_service


PARENT_PATH = os.getcwd()


class YoloDetect:
    def __init__(self, user_id, image):
        save_path = Constants.PUBLIC_FOLDER + image

        self.source = os.path.join(PARENT_PATH, save_path)

        self.user_id = user_id
        # self.weights = ROOT / "best_vbase.pt"
        self.weights = ROOT / "best.pt"
        self.data = ROOT / "datasets/data.yaml"
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.dnn = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = 0
        self.agnostic_nms = False
        self.max_det = 1000
        self.augment = False
        self.visualize = False
        self.exist_ok = False
        self.vid_stride = 1
        self.check = False

    def alert_image(self, file_name):
        try:
            response = requests.post(
                f"http://{env_service.get_env_var('BASE_ADDRESS')}:8008/api/notification",
                json={"user_id": str(self.user_id), "detection_path": file_name},
            )
            if response.status_code == 200:
                print("Request successful")
                print("Response JSON:", response.json())
            else:
                print(f"Request failed with status code {response.status_code}")
                print("Response text:", response.text)
        except Exception as e:
            print(e)

    def detection_thread(self):
        if (
            self.source.endswith(".png")
            or self.source.endswith(".jpg")
            or self.source.endswith(".jpeg")
        ):
            img = cv2.imread(self.source)
            img, checked, file_name = self.detect_single_image(img)

            response = requests.post(
                f"http://{env_service.get_env_var('BASE_ADDRESS')}:8008/api/history",
                json={
                    "user_id": str(self.user_id),
                    "result_path": file_name,
                    "status": checked,
                },
            )
            if response.status_code == 200 or response.status_code == 201:
                print("Request successful")
                print("Response JSON:", response.json())
            else:
                print(f"Request failed with status code {response.status_code}")
                print("Response text:", response.text)
        else:
            cap = cv2.VideoCapture(self.source)
            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    else:
                        img, attr = self.detect_image(frame)
                        cv2.imshow("", img)
                        cv2.waitKey(0)
            except Exception as e:
                print(e)
                pass
            finally:
                cap.release()
                cv2.destroyAllWindows()

    def detect_single_image(self, img):
        source = str(self.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        if is_file:
            source = check_file(source)  # download

        # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(
            self.weights, device=device, dnn=self.dnn, data=self.data, fp16=False
        )
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        im = im0s = img
        im = letterbox(im, self.imgsz, stride=stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=self.augment, visualize=self.visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                self.classes,
                self.agnostic_nms,
                max_det=self.max_det,
            )

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            im0 = im0s.copy()

            s = "%gx%g " % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                self.check = True
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f"{names[c]}"

                    c = int(cls)  # integer class
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()

            # Print time (inference-only)
            LOGGER.info(
                f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms"
            )

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(
            f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
            % t
        )

        now = datetime.datetime.now()
        file_name = str(int(datetime.datetime.timestamp(now))) + ".jpg"
        save_path = os.path.join(PARENT_PATH, Constants.DETECTION_FOLDER + file_name)
        cv2.imwrite(save_path, im0)

        if self.check == True:
            # Save results (image with detections)
            self.alert_image(file_name)

        return im0, self.check, file_name

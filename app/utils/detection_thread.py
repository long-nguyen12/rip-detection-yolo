import os
import threading
from yolov5.yolo_detect import YoloDetect


class DetectionThread:
    def __init__(self) -> None:
        pass

    def check_thread_by_name(self, name):
        threads = threading.enumerate()
        for thread in threads:
            if thread.name == name:
                return True
        return False

    def get_thread_by_name(self, name):
        threads = threading.enumerate()
        for thread in threads:
            if thread.name == name:
                return thread

    def start_detecting(self, user_id, video):
        detection_task = YoloDetect(user_id, video)
        thread = threading.Thread(target=detection_task.detection_thread, name=video)
        thread.start()

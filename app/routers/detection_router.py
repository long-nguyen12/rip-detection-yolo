from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from app.schemas import detection
from app.config.mongo_service import db_mongo
from app.middlewares.auth_bearer import JwtBearer
from app.services.detection_service import DetectionService
from app.services.user_service import UserService
from app.utils.detection_thread import DetectionThread
from yolov5.yolo_detect import YoloDetect

router = APIRouter()
detection_service = DetectionService()
user_service = UserService()


@router.post("/api/detection/", dependencies=[Depends(JwtBearer())], tags=["detections"])
async def create_detection(
    detection: detection.DetectionSchema = Body(...), username=Depends(JwtBearer())
):
    detection_dict = jsonable_encoder(detection)
    detection_data = await detection_service.create(detection_dict)
    user = await user_service.get("username", username)
    detection_thread = DetectionThread()
    detection_thread.start_detecting(
        user.id,
        detection.source,
    )
    # detection_task = YoloDetect(user.id, detection.source)
    # await detection_task.detection_thread()

    return detection_data

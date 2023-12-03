from pydantic import BaseModel, Field
from app.schemas.base import PyObjectId
from bson import ObjectId


class DetectionSchema(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    source: str = Field(...)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {"example": {}}

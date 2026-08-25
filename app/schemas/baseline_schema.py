from pydantic import BaseModel, Field


class Phone(BaseModel):
    model: str = Field(description="The model of the phone")
    storage: str = Field(description="The storage of the phone")
    camera: str = Field(description="The camera of the phone")
    battery: str = Field(description="The battery of the phone")
    year: int = Field(description="The year of the phone")

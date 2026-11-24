from pydantic import BaseModel, Field


class Phone(BaseModel):
    model: str | None = Field(default=None, description="The model of the phone")
    storage: str | None = Field(default=None, description="The storage of the phone")
    camera: str | None = Field(default=None, description="The camera of the phone")
    battery: str | None = Field(default=None, description="The battery of the phone")
    year: int | None = Field(default=None, description="The year of the phone")

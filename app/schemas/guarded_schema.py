from pydantic import BaseModel, Field
from guardrails import register_validator, Validator
from guardrails.hub import ValidLength, ValidRange
from guardrails.validator_base import FailResult, PassResult


class GuardedPhone(BaseModel):

    @register_validator("valid_phone_storage", data_type="string")
    class ValidPhoneStorage(Validator):
        """Validates that storage is a realistic phone storage size."""

        def __init__(self, on_fail=None):
            super().__init__(on_fail=on_fail)
            self.valid_storages = {"64GB", "128GB", "256GB", "512GB", "1TB"}

        def validate(self, value, metadata):
            if value.strip() not in self.valid_storages:
                return FailResult(
                    error_message=f"Storage must be one of {self.valid_storages}, got '{value}'"
                )
            return PassResult()

    model: str = Field(
        description="The model of the phone",
        validators=[ValidLength(min=2, max=50)],
    )
    storage: str = Field(
        description="The storage of the phone (64GB, 128GB, 256GB, 512GB, or 1TB)",
        validators=[ValidPhoneStorage()],
    )
    camera: str = Field(
        description="The camera of the phone",
        validators=[ValidLength(min=2, max=100)],
    )
    battery: str = Field(
        description="The battery of the phone",
        validators=[ValidLength(min=2, max=50)],
    )
    year: int = Field(
        description="The year of the phone",
        validators=[ValidRange(min=2000, max=2030)],
    )

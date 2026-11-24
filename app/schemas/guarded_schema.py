from pydantic import BaseModel, Field
from guardrails import register_validator, Validator
from guardrails.hub import ValidLength, ValidRange
from guardrails.validator_base import FailResult, PassResult


def _allow_none(validator_cls):
    """Wrap a hub validator so a null field (not in the source) passes.

    Guardrails still runs validators on explicit nulls, and ValidLength /
    ValidRange crash on None. Non-null values are validated unchanged.
    """

    class AllowNone(validator_cls):
        def validate(self, value, metadata):
            if value is None:
                return PassResult()
            return super().validate(value, metadata)

    AllowNone.__name__ = validator_cls.__name__
    return AllowNone


ValidLength = _allow_none(ValidLength)
ValidRange = _allow_none(ValidRange)


class GuardedPhone(BaseModel):

    @register_validator("valid_phone_storage", data_type="string")
    class ValidPhoneStorage(Validator):
        """Validates that storage is a realistic phone storage size."""

        def __init__(self, on_fail=None):
            super().__init__(on_fail=on_fail)
            self.valid_storages = {"64GB", "128GB", "256GB", "512GB", "1TB"}

        def validate(self, value, metadata):
            if value is None:
                return PassResult()
            if value.strip() not in self.valid_storages:
                return FailResult(
                    error_message=f"Storage must be one of {self.valid_storages}, got '{value}'"
                )
            return PassResult()

    model: str | None = Field(
        default=None,
        description="The model of the phone",
        json_schema_extra={"validators": [ValidLength(min=2, max=50)]},
    )
    storage: str | None = Field(
        default=None,
        description="The storage of the phone (64GB, 128GB, 256GB, 512GB, or 1TB)",
        json_schema_extra={"validators": [ValidPhoneStorage()]},
    )
    camera: str | None = Field(
        default=None,
        description="The camera of the phone",
        json_schema_extra={"validators": [ValidLength(min=2, max=100)]},
    )
    battery: str | None = Field(
        default=None,
        description="The battery of the phone",
        json_schema_extra={"validators": [ValidLength(min=2, max=50)]},
    )
    year: int | None = Field(
        default=None,
        description="The year of the phone",
        json_schema_extra={"validators": [ValidRange(min=2000, max=2030)]},
    )

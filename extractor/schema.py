"""
Pydantic models for weld log records.

We normalise and validate input fields and provide simple validators to
enforce consistent formatting, e.g., W123 becomes W-123 and ND 25 becomes
uppercase. Additional validators can be added as needed (e.g., range
checks).
"""

from typing import Optional, List
from pydantic import BaseModel, field_validator, ConfigDict


class WeldRecord(BaseModel):
    weld_number: Optional[str] = None
    joint_size: Optional[str] = None
    joint_type: Optional[str] = None
    material_description: Optional[str] = None
    source_file: Optional[str] = None
    source_page: Optional[int] = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("weld_number")
    @classmethod
    def norm_weld_number(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        val = v.strip().upper()
        if val.startswith("W") and "-" not in val and val[1:].isdigit():
            val = f"W-{val[1:]}"
        return val

    @field_validator("joint_size")
    @classmethod
    def norm_joint_size(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        return v.strip().upper().replace("  ", " ")

    @field_validator("joint_type")
    @classmethod
    def norm_joint_type(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        return v.strip().upper()


class WeldLog(BaseModel):
    rows: List[WeldRecord]

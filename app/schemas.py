from pydantic import BaseModel, Field

class PredictInput(BaseModel):
    air_temperature_k: float = Field(..., alias="Air temperature [K]")
    process_temperature_k: float = Field(..., alias="Process temperature [K]")
    rotational_speed_rpm: float = Field(..., alias="Rotational speed [rpm]")
    torque_nm: float = Field(..., alias="Torque [Nm]")
    tool_wear_min: float = Field(..., alias="Tool wear [min]")
    type_h: int = 0
    type_m: int = 0

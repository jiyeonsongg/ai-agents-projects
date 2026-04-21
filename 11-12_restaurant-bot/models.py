from pydantic import BaseModel
from typing import Optional


class UserAccountContext(BaseModel):

    customer_id: int
    name: str
    tier: str = "basic"
    email: Optional[str] = None

    def is_premium_customer(self) -> bool:
        t = (self.tier or "").lower()
        return t not in ("basic", "standard", "")

    def add_troubleshooting_step(self, step: str) -> None:
        pass


class InputGuardRailOutput(BaseModel):

    is_off_topic: bool
    reason: str


class RestaurantOutputGuardRailOutput(BaseModel):

    contains_inappropriate: bool
    contains_off_topic_for_agent: bool
    reason: str


class HandoffData(BaseModel):

    to_agent_name: str
    issue_type: str
    issue_description: str
    reason: str
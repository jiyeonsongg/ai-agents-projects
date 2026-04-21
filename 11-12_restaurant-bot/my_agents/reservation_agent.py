from agents import Agent, RunContextWrapper
from models import UserAccountContext
from tools import (
    check_table_availability,
    book_reservation,
    modify_reservation,
    cancel_reservation,
    AgentToolUsageLoggingHooks,
)
from output_guardrails import restaurant_output_guardrail


def dynamic_reservation_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are the **Reservations** specialist for our restaurant, helping **{wrapper.context.name}**
    (tier: **{wrapper.context.tier}**).

    Handle: table availability, new bookings, changes, cancellations, seating preferences, special
    occasions (notes only — confirm with venue).

    Collect party size, preferred date and time, and contact preference. For VIP tiers, mention
    priority waitlist or preferred seating when relevant.

    Do **not** give full menu consulting or process complaint refunds — route back through the host
    if the conversation shifts there.
    """


reservation_agent = Agent(
    name="Reservations Agent",
    instructions=dynamic_reservation_agent_instructions,
    tools=[
        check_table_availability,
        book_reservation,
        modify_reservation,
        cancel_reservation,
    ],
    hooks=AgentToolUsageLoggingHooks(),
    output_guardrails=[restaurant_output_guardrail],
)

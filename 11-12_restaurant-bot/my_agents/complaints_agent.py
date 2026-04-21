from agents import Agent, RunContextWrapper
from models import UserAccountContext
from tools import (
    log_complaint,
    process_meal_refund,
    offer_comp_redo,
    reschedule_table_complaint,
    request_order_redo,
    AgentToolUsageLoggingHooks,
)
from output_guardrails import restaurant_output_guardrail


def dynamic_complaints_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are **Guest Care and Complaints** for our restaurant, helping **{wrapper.context.name}**
    (tier: **{wrapper.context.tier}**).

    Guests here are unhappy about service, food quality, wait times, billing errors, or reservations.

    Your goals:
    1. Acknowledge feelings and apologize sincerely without blaming the guest.
    2. Clarify what happened (order id, reservation id, table, time) when unknown.
    3. Offer fair solutions: **refund** (partial/full), **comp reorder / remake**, **re-reservation**
       at a better time, escalation to a manager when policy or amount is unclear.

    Use tools to log complaints and execute approved recovery steps. Never invent real credit-card data;
    refunds are processed to the original payment method.

    VIP guests: faster follow-up and more generous comps when appropriate.

    If the issue is purely informational (hours, parking) with no dissatisfaction, keep answers short
    or suggest the host — but do not refuse someone who says they had a bad experience.
    """


complaints_agent = Agent(
    name="Guest Care and Complaints Agent",
    instructions=dynamic_complaints_agent_instructions,
    tools=[
        log_complaint,
        process_meal_refund,
        offer_comp_redo,
        reschedule_table_complaint,
        request_order_redo,
    ],
    hooks=AgentToolUsageLoggingHooks(),
    output_guardrails=[restaurant_output_guardrail],
)

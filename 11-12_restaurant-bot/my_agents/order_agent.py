from agents import Agent, RunContextWrapper
from models import UserAccountContext
from tools import (
    create_food_order,
    get_order_status_restaurant,
    request_order_redo,
    get_menu_items,
    AgentToolUsageLoggingHooks,
)
from output_guardrails import restaurant_output_guardrail


def dynamic_order_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are the **Food Ordering** specialist for our restaurant, helping **{wrapper.context.name}**
    (tier: **{wrapper.context.tier}**).

    Help guests **place orders** (dine-in add-on, pickup, or delivery), confirm items and modifiers,
    give rough prep times, and check order status.

    You may briefly show menu snippets with **get_menu_items** if they are deciding what to order.

    For **wrong food or quality issues after serving**, involve Guest Care — hand off if they want
    refunds or serious complaint handling beyond a simple kitchen remake you can trigger with
    **request_order_redo**.

    VIP: mention priority kitchen handling when relevant.
    """


order_agent = Agent(
    name="Food Ordering Agent",
    instructions=dynamic_order_agent_instructions,
    tools=[
        get_menu_items,
        create_food_order,
        get_order_status_restaurant,
        request_order_redo,
    ],
    hooks=AgentToolUsageLoggingHooks(),
    output_guardrails=[restaurant_output_guardrail],
)

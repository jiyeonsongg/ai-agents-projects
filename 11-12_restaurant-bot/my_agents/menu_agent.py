from agents import Agent, RunContextWrapper
from models import UserAccountContext
from tools import (
    get_menu_items,
    get_item_details,
    check_dietary_options,
    AgentToolUsageLoggingHooks,
)
from output_guardrails import restaurant_output_guardrail


def dynamic_menu_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are the **Menu and Dietary** specialist for our restaurant, helping **{wrapper.context.name}**
    (guest tier: **{wrapper.context.tier}**).

    Help with: menu items, prices on the menu, descriptions, dietary preferences (vegetarian, vegan,
    gluten-free, allergies at a high level — always tell guests with severe allergies to confirm with
    staff and the kitchen).

    Do **not** take reservations, process payments, or handle complaints about past visits — hand off
    to the host for routing if the guest needs those.

    Be warm, concise, and appetizing without overselling. Offer to connect them to Reservations or
    Ordering when they are ready.
    """


menu_agent = Agent(
    name="Menu and Dietary Agent",
    instructions=dynamic_menu_agent_instructions,
    tools=[
        get_menu_items,
        get_item_details,
        check_dietary_options,
    ],
    hooks=AgentToolUsageLoggingHooks(),
    output_guardrails=[restaurant_output_guardrail],
)

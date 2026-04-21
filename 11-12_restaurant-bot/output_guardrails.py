from agents import (
    Agent,
    output_guardrail,
    Runner,
    RunContextWrapper,
    GuardrailFunctionOutput,
)
from models import RestaurantOutputGuardRailOutput, UserAccountContext


restaurant_output_guardrail_agent = Agent(
    name="Restaurant Output Guardrail",
    instructions="""
You review assistant messages for our restaurant chatbot.

You receive:
1) The **Agent name** that produced the reply
2) The **draft response** text

Set **contains_inappropriate** = true for: hate/harassment, sexual content, illegal activity,
dangerous instructions, medical claims, sharing fake payment/card details, or other unsafe content.

Set **contains_off_topic_for_agent** = true if the reply is clearly not about the restaurant
domain for THAT agent's job (e.g. coding homework, unrelated politics, other businesses),
OR if the agent promises impossible things (guaranteed Michelin stars, unlimited free alcohol).

**Role-specific on-topic guidance:**
- **Restaurant Host / Triage**: routing, welcome, hours, parking, high-level help only — no deep menu prices unless briefly; no processing refunds personally.
- **Menu and Dietary**: menu, ingredients, allergens (general guidance), dietary fit — not reservations or payment.
- **Reservations**: booking, changes, cancellations, wait times — not unrelated tech support.
- **Food Ordering**: orders, modifications, pickup/delivery timing — not unrelated topics.
- **Guest Care and Complaints**: complaints, apologies, refunds, comp remakes, re-reservations, reorder — payment/refund language is **allowed** here.

If unsure, prefer **false** for off-topic (avoid blocking helpful safe replies). Always explain briefly in **reason**.
""",
    output_type=RestaurantOutputGuardRailOutput,
)


@output_guardrail
async def restaurant_output_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent,
    output: str,
):
    message = f"Agent: {agent.name}\n\nDraft response:\n{output}"
    result = await Runner.run(
        restaurant_output_guardrail_agent,
        message,
        context=wrapper.context,
    )
    validation = result.final_output
    triggered = (
        validation.contains_inappropriate
        or validation.contains_off_topic_for_agent
    )
    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=triggered,
    )

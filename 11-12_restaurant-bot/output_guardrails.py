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

Set **contains_off_topic_for_agent** = true ONLY when the reply is clearly unrelated to the
restaurant domain (e.g. coding homework, unrelated politics, other businesses), OR when it
promises impossible things (guaranteed Michelin stars, unlimited free alcohol).

**Role-specific on-topic guidance:**
- **Restaurant Host / Triage**: routing, welcome, hours, parking, and quick helpful answers are allowed
  (including brief menu help) before handoff.
- **Menu and Dietary**: menu, ingredients, allergens (general guidance), dietary fit, and prices.
- **Reservations / Reservations Agent**: table availability, booking intent, collecting party size/date/time,
  seating preferences, special occasions, waitlist guidance, changes, cancellations, and wait times.
- **Food Ordering**: orders, modifications, pickup/delivery timing, and menu suggestions tied to ordering.
- **Guest Care and Complaints**: complaints, apologies, refunds, comp remakes, re-reservations, reorder.

Do NOT mark off-topic just because another specialist could answer better; that is a routing quality
issue, not a safety issue.

If unsure, prefer **false** for off-topic (avoid blocking helpful safe replies). Use **true** only for
clear mismatches. Always explain briefly in **reason**.
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

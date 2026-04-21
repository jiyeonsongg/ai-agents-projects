import streamlit as st
from agents import function_tool, AgentHooks, Agent, Tool, RunContextWrapper
from typing import Optional
from models import UserAccountContext
import random
from datetime import datetime, timedelta


# =============================================================================
# MENU & DIETARY
# =============================================================================


@function_tool
def get_menu_items(
    context: UserAccountContext, category: str = "all"
) -> str:
    """Return menu items. category: appetizers, mains, desserts, drinks, or all."""
    menu = {
        "appetizers": [
            "Bruschetta Trio — $12",
            "Calamari Fritti — $15",
            "Seasonal Soup — $9",
        ],
        "mains": [
            "Grilled Salmon — $28",
            "Herb Roasted Chicken — $24",
            "Wild Mushroom Risotto — $22 (vegetarian)",
            "Ribeye 12oz — $42",
        ],
        "desserts": [
            "Tiramisu — $10",
            "Lemon Sorbet — $7 (vegan)",
        ],
        "drinks": [
            "House Red / White — $9",
            "Sparkling Water — $4",
            "Espresso — $4",
        ],
    }
    cat = category.lower().strip()
    if cat == "all":
        lines = [f"**{k.title()}**\n" + "\n".join(v) for k, v in menu.items()]
        return "📋 **Today's menu**\n\n" + "\n\n".join(lines)
    if cat in menu:
        return f"📋 **{cat.title()}**\n" + "\n".join(menu[cat])
    return f"No category '{category}'. Try: appetizers, mains, desserts, drinks, all."


@function_tool
def get_item_details(context: UserAccountContext, item_name: str) -> str:
    """Allergens, dietary tags, and short description for a dish or drink."""
    return f"""
🍽️ **{item_name}**
• Contains: gluten, dairy (example — confirm with kitchen for severe allergies)
• Chef's note: Made to order; substitutions may be limited at peak hours.
• Ask your server to flag nut or shellfish allergies before ordering.
    """.strip()


@function_tool
def check_dietary_options(
    context: UserAccountContext, restriction: str
) -> str:
    """Suggest options for vegetarian, vegan, gluten-free, nut-free, etc."""
    r = restriction.lower()
    hints = {
        "vegetarian": "Mushroom risotto, seasonal soup (no bacon), several sides.",
        "vegan": "Lemon sorbet, modified bruschetta without cheese (ask kitchen).",
        "gluten-free": "Salmon without croutons; risotto can be GF on request.",
        "nut-free": "Kitchen can prepare nut-free; always disclose severity.",
    }
    return hints.get(
        r,
        f"We can usually accommodate **{restriction}**. Please confirm with the host or server.",
    )


# =============================================================================
# RESERVATIONS
# =============================================================================


@function_tool
def check_table_availability(
    context: UserAccountContext, party_size: int, date_time: str
) -> str:
    """Check if tables are free for a party on a requested date/time string."""
    ok = party_size <= 8 and random.random() > 0.15
    if ok:
        return f"""
✅ Tables likely available for **{party_size}** on **{date_time}**
• Suggested slots: 5:30 PM, 7:00 PM, 8:30 PM
• Patio seating: weather permitting
        """.strip()
    return f"""
⚠️ Limited availability for **{party_size}** on **{date_time}**
• Next openings: 9:15 PM bar seating, or **{ (datetime.now() + timedelta(days=1)).strftime('%A') }** dinner
    """.strip()


@function_tool
def book_reservation(
    context: UserAccountContext,
    party_size: int,
    date_time: str,
    seating_preference: str = "no preference",
) -> str:
    """Confirm a table reservation."""
    rid = f"RES-{random.randint(10000, 99999)}"
    return f"""
✅ Reservation confirmed
🔖 Confirmation: **{rid}**
👥 Party: {party_size}
📅 **{date_time}**
🪑 Preference: {seating_preference}
📧 Sent to: {context.email or 'on file'}
    """.strip()


@function_tool
def modify_reservation(
    context: UserAccountContext,
    reservation_id: str,
    new_date_time: str,
    party_size: Optional[int] = None,
) -> str:
    """Change time or party size for an existing reservation."""
    ps = f"\n👥 New party size: {party_size}" if party_size else ""
    return f"""
✅ Reservation updated
🔖 **{reservation_id}**
📅 New time: **{new_date_time}**{ps}
    """.strip()


@function_tool
def cancel_reservation(
    context: UserAccountContext, reservation_id: str, reason: str = ""
) -> str:
    """Cancel a reservation."""
    return f"""
✅ Cancellation recorded
🔖 **{reservation_id}**
📝 Note: {reason or 'Guest requested'}
We hope to welcome you again soon.
    """.strip()


# =============================================================================
# FOOD ORDERS (takeout / dine-in add-on)
# =============================================================================


@function_tool
def create_food_order(
    context: UserAccountContext,
    items: str,
    order_type: str = "dine_in",
    special_requests: str = "",
) -> str:
    """Place an order. items: comma-separated names. order_type: dine_in, pickup, delivery."""
    oid = f"ORD-{random.randint(100000, 999999)}"
    eta = "25–35 min" if order_type == "pickup" else "15–25 min (kitchen)"
    return f"""
✅ Order placed
🔖 **{oid}**
🍽️ Items: {items}
📦 Type: {order_type.replace('_', ' ')}
📝 Requests: {special_requests or 'None'}
⏱️ ETA: {eta}
📧 {context.email or 'Pickup name at counter'}
    """.strip()


@function_tool
def get_order_status_restaurant(
    context: UserAccountContext, order_id: str
) -> str:
    """Status for a restaurant order id."""
    status = random.choice(["received", "in_kitchen", "ready", "out_for_delivery"])
    return f"📦 **{order_id}** — status: **{status}**"


@function_tool
def request_order_redo(
    context: UserAccountContext, order_id: str, issue: str
) -> str:
    """Kitchen remake for wrong or unsatisfactory dish (coordinate with complaints if needed)."""
    return f"""
🔄 Remake requested for **{order_id}**
📝 Issue: {issue}
Priority queue: {'yes (VIP)' if context.is_premium_customer() else 'standard'}
    """.strip()


# =============================================================================
# COMPLAINTS & RECOVERY
# =============================================================================


@function_tool
def log_complaint(
    context: UserAccountContext,
    summary: str,
    severity: str = "medium",
) -> str:
    """Record a guest complaint for management follow-up."""
    cid = f"CMP-{random.randint(10000, 99999)}"
    return f"""
📋 Complaint logged — **{cid}**
📝 {summary}
⚡ Severity: {severity}
Manager will review within {'1 hour' if context.is_premium_customer() else '24 hours'}.
    """.strip()


@function_tool
def process_meal_refund(
    context: UserAccountContext,
    order_or_check_id: str,
    amount: float,
    reason: str,
) -> str:
    """Issue a refund for a meal or check (after complaint review)."""
    rid = f"REF-{random.randint(100000, 999999)}"
    days = 2 if context.is_premium_customer() else 5
    return f"""
✅ Refund approved
🔗 **{rid}** for **{order_or_check_id}**
💰 ${amount:.2f}
📝 {reason}
⏱️ {days} business days to original payment
    """.strip()


@function_tool
def offer_comp_redo(
    context: UserAccountContext,
    original_order_id: str,
    replacement_items: str,
) -> str:
    """Complimentary replacement order."""
    return f"""
🎁 Comp reorder for **{original_order_id}**
🍽️ Replacement: {replacement_items}
No charge — please allow kitchen standard prep time.
    """.strip()


@function_tool
def reschedule_table_complaint(
    context: UserAccountContext,
    reservation_id: str,
    new_slot: str,
    note: str = "",
) -> str:
    """Re-book after a bad experience (re-reservation recovery)."""
    return f"""
✅ Re-reservation set
🔖 **{reservation_id}** → **{new_slot}**
📝 {note or 'Guest recovery'}
    """.strip()


class AgentToolUsageLoggingHooks(AgentHooks):

    async def on_tool_start(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        tool: Tool,
    ):
        with st.sidebar:
            st.write(f"🔧 **{agent.name}** starting tool: `{tool.name}`")

    async def on_tool_end(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        tool: Tool,
        result: str,
    ):
        with st.sidebar:
            st.write(f"🔧 **{agent.name}** used tool: `{tool.name}`")
            st.code(result)

    async def on_handoff(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        source: Agent[UserAccountContext],
    ):
        with st.sidebar:
            st.write(f"🔄 Handoff: **{source.name}** → **{agent.name}**")

    async def on_start(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
    ):
        with st.sidebar:
            st.write(f"🚀 **{agent.name}** activated")

    async def on_end(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        output,
    ):
        with st.sidebar:
            st.write(f"🏁 **{agent.name}** completed")

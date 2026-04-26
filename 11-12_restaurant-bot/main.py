import dotenv

dotenv.load_dotenv()
import asyncio
import os
import streamlit as st
from agents import (
    Runner,
    SQLiteSession,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
from models import UserAccountContext
from my_agents.triage_agent import triage_agent

# Streamlit deployment: inject key from secrets.toml into env var expected by SDK.
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

user_account_ctx = UserAccountContext(
    customer_id=1,
    name="Jinni",
    tier="VIP",
    email="jinni@example.com",
)

st.set_page_config(page_title="Restaurant bot", page_icon="🍽️")

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "restaurant-bot-memory.db",
    )
session = st.session_state["session"]

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent

if "messages" not in st.session_state:
    st.session_state["messages"] = []


async def run_turn(user_text: str):
    return await Runner.run(
        st.session_state["agent"],
        user_text,
        context=user_account_ctx,
        session=session,
    )


st.title("Restaurant assistant")
st.caption("Menu, reservations, ordering, and guest care — powered by your agents.")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the menu, book a table, place an order…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = asyncio.run(run_turn(prompt))
            except InputGuardrailTripwireTriggered:
                reply = (
                    "I can only help with our restaurant: menu, reservations, orders, "
                    "hours, or guest care. What would you like to know?"
                )
            except OutputGuardrailTripwireTriggered:
                reply = (
                    "I cannot share that reply. Please rephrase your question about "
                    "the restaurant."
                )
            except Exception as e:
                st.error(f"Something went wrong: {e!s}")
                reply = "Sorry — please try again in a moment."
            else:
                out = result.final_output
                reply = out if isinstance(out, str) else str(out)
                st.session_state["agent"] = result.last_agent

        st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
        st.session_state["messages"] = []
        st.session_state["agent"] = triage_agent
        st.rerun()
    with st.expander("Session debug"):
        st.write(asyncio.run(session.get_items()))

import asyncio
import streamlit as st
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
)

try:
    # Optional convenience if you use a local `.env` for OPENAI_API_KEY.
    import dotenv  # type: ignore

    dotenv.load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Life Coach Agent", page_icon="💬")
st.title("Life Coach Agent")

if "session" not in st.session_state:
    # Persists chat history to a local sqlite db file.
    st.session_state["session"] = SQLiteSession("chat-history", "chat-memory.db")
session: SQLiteSession = st.session_state["session"]

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach Agent",
        instructions=(
            "You are a helpful life coach and assistant.\n"
            "Use web search when the user asks for up-to-date information or sources.\n"
            "When you use web search, summarize clearly and cite where the info came from."
        ),
        tools=[WebSearchTool()],
    )
agent: Agent = st.session_state["agent"]


async def paint_history():
    messages = await session.get_items()

    for message in messages:
        role = message.get("role")
        if role in ("user", "assistant", "ai"):
            # Normalize to Streamlit roles.
            st_role = "assistant" if role in ("assistant", "ai") else "user"
            with st.chat_message(st_role):
                content = message.get("content")
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, list):
                    # Agents SDK may store structured content; render text parts.
                    for part in content:
                        text = part.get("text") if isinstance(part, dict) else None
                        if text:
                            st.write(str(text))
        elif message.get("type") == "web_search_call":
            with st.chat_message("assistant"):
                st.write("Searched the web…")


asyncio.run(paint_history())


async def run_agent(message):
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        text_placeholder = st.empty()
        response_text = ""

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type != "raw_response_event":
                continue

            # Stream assistant text.
            if event.data.type == "response.output_text.delta":
                response_text += event.data.delta
                text_placeholder.write(response_text)

            # Keep the user informed about web search progress.
            elif event.data.type == "response.web_search_call.in_progress":
                status.update(label="Searching the web…", state="running")
            elif event.data.type == "response.web_search_call.searching":
                status.update(label="Searching the web…", state="running")
            elif event.data.type == "response.web_search_call.completed":
                status.update(label="Web search completed.", state="complete")
            elif event.data.type == "response.completed":
                status.update(label="Done.", state="complete")


prompt = st.chat_input("Message")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    asyncio.run(run_agent(prompt))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
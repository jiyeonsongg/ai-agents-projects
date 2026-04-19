import asyncio
import base64
import os
import tempfile
from io import BytesIO

import streamlit as st
from agents import (
    Agent,
    FileSearchTool,
    ImageGenerationTool,
    Runner,
    SQLiteSession,
    WebSearchTool,
)
from openai import OpenAI

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

if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = OpenAI()
if "planner_vector_store_id" not in st.session_state:
    st.session_state["planner_vector_store_id"] = None
if "planner_source_names" not in st.session_state:
    st.session_state["planner_source_names"] = []
if "indexed_files_fingerprint" not in st.session_state:
    st.session_state["indexed_files_fingerprint"] = ""

client: OpenAI = st.session_state["openai_client"]


def image_from_b64(b64_data: str) -> BytesIO:
    return BytesIO(base64.b64decode(b64_data))


def get_or_create_vector_store_id() -> str:
    current_id = st.session_state.get("planner_vector_store_id")
    if current_id:
        return current_id
    vector_store = client.vector_stores.create(name="life-coach-planner-store")
    st.session_state["planner_vector_store_id"] = vector_store.id
    return vector_store.id


def index_uploaded_files(uploaded_files) -> tuple[int, str]:
    vector_store_id = get_or_create_vector_store_id()
    temp_paths: list[str] = []
    files_to_upload = []
    try:
        for uploaded in uploaded_files:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded.getvalue())
                temp_paths.append(tmp_file.name)
            files_to_upload.append(open(temp_paths[-1], "rb"))
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=files_to_upload,
        )
    finally:
        for handle in files_to_upload:
            handle.close()
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
    return len(uploaded_files), vector_store_id


def build_agent() -> Agent:
    tools = [
        WebSearchTool(),
        ImageGenerationTool(
            tool_config={
                "type": "image_generation",
                "model": "gpt-image-1",
                "quality": "high",
                "size": "1536x1024",
                "partial_images": 3,
            }
        ),
    ]
    vector_store_id = st.session_state.get("planner_vector_store_id")
    if vector_store_id:
        tools.append(
            FileSearchTool(
                vector_store_ids=[vector_store_id],
                max_num_results=4,
            )
        )
    return Agent(
        name="Life Coach Agent",
        model="gpt-5",
        instructions=(
            "You are a helpful life coach and assistant.\n\n"
            "Tools and when to use them:\n"
            "- FileSearchTool: When planner or diary files are indexed, search them first for "
            "the user's stated goals, habits, constraints, and past reflections. Ground your "
            "coaching in what they actually wrote.\n"
            "- WebSearchTool: Use for current tips, research-backed advice, examples, or "
            "motivational content from the wider world. Summarize clearly and cite sources.\n"
            "- ImageGenerationTool: Use when the user wants a visual—especially a goals-oriented "
            "vision board or a motivational message poster. Before generating, briefly align on "
            "their goal (or infer from context and FileSearchTool). Craft a single detailed "
            "image prompt: layout, colors, typography style for any text on the poster, symbols "
            "that match their goal, and mood (uplifting, calm, bold). For vision boards, describe "
            "a clear collage or grid with distinct zones per goal. For posters, include the "
            "exact short headline or quote they want (or you propose one sentence max). "
            "Generate one image per request unless they explicitly ask for multiple.\n\n"
            "Personalize advice using this chat and prior assistant turns. After generating an "
            "image, briefly explain how it reflects their goals; do not repeat the full prompt."
        ),
        tools=tools,
    )


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


async def build_memory_context() -> str:
    items = await session.get_items()
    prior_advice: list[str] = []
    for item in items:
        role = item.get("role")
        if role not in ("assistant", "ai"):
            continue
        content = item.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]))
            text = " ".join(text_parts)
        else:
            text = ""
        cleaned = text.strip()
        if cleaned:
            prior_advice.append(cleaned)
    if not prior_advice:
        return ""
    return "\n\n".join(prior_advice[-3:])


async def run_agent(message):
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        text_placeholder = st.empty()
        image_area = st.empty()
        response_text = ""

        memory_context = await build_memory_context()
        planner_list = ", ".join(st.session_state.get("planner_source_names", []))
        contextual_message = message
        if memory_context:
            contextual_message = (
                f"{message}\n\n"
                "Previous assistant advice for personalization:\n"
                f"{memory_context}\n"
            )
        if planner_list:
            contextual_message += (
                "\nUploaded planner files available for FileSearchTool: "
                f"{planner_list}"
            )

        stream = Runner.run_streamed(build_agent(), contextual_message, session=session)

        async for event in stream.stream_events():
            if event.type != "raw_response_event":
                continue

            event_type = getattr(event.data, "type", "")

            # Stream assistant text.
            if event_type == "response.output_text.delta":
                response_text += event.data.delta
                text_placeholder.write(response_text)

            # Keep the user informed about tool progress.
            elif event_type in (
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
            ):
                status.update(label="📲Searching the web…", state="running")
            elif "file_search_call" in event_type and (
                "in_progress" in event_type or "searching" in event_type
            ):
                status.update(label="👓Searching planner file…", state="running")
            elif "file_search_call" in event_type and "completed" in event_type:
                status.update(label="🪄Planner file search completed.", state="running")
            elif event_type == "response.web_search_call.completed":
                status.update(label="🪄Web search completed.", state="complete")
            elif event_type in (
                "response.image_generation_call.in_progress",
                "response.image_generation_call.generating",
            ):
                status.update(label="🎰Generating image…", state="running")
            elif event_type == "response.image_generation_call.partial_image":
                status.update(label="🎰Generating image…", state="running")
                b64 = getattr(event.data, "partial_image_b64", "") or ""
                if b64:
                    try:
                        image_area.image(image_from_b64(b64))
                    except (ValueError, OSError):
                        pass
            elif event_type == "response.image_generation_call.completed":
                status.update(label="🪄Image generation finished.", state="running")
            elif event_type == "response.output_item.done":
                item = getattr(event.data, "item", None)
                if item is not None and getattr(item, "type", None) == "image_generation_call":
                    result = getattr(item, "result", None)
                    if result:
                        try:
                            image_area.image(image_from_b64(result))
                        except (ValueError, OSError):
                            pass
            elif event_type == "response.completed":
                status.update(label="Done.", state="complete")


prompt = st.chat_input("Message")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    asyncio.run(run_agent(prompt))


with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload planner (.pdf or .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        fingerprint = "|".join(f"{f.name}:{f.size}" for f in uploaded_files)
        if fingerprint != st.session_state.get("indexed_files_fingerprint"):
            if st.button("Index planner files"):
                try:
                    indexed_count, vector_store_id = index_uploaded_files(uploaded_files)
                    st.session_state["planner_source_names"] = [f.name for f in uploaded_files]
                    st.session_state["indexed_files_fingerprint"] = fingerprint
                    st.success(
                        f"Indexed {indexed_count} file(s) with SDK FileSearchTool "
                        f"(vector store: {vector_store_id})."
                    )
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
        else:
            st.info("These files are already indexed.")

    current_sources: list[str] = st.session_state.get("planner_source_names", [])
    if current_sources:
        st.caption("Indexed planner files:")
        for source in current_sources:
            st.write(f"- {source}")

    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
        st.session_state["planner_vector_store_id"] = None
        st.session_state["planner_source_names"] = []
        st.session_state["indexed_files_fingerprint"] = ""
"""Google ADK multi-agent app for a children's story book maker."""

from __future__ import annotations

import asyncio
import json
import os
import re
from base64 import b64decode
from typing import Any, AsyncGenerator, Literal

import requests
from google.adk.agents import BaseAgent, Context, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.models.llm_response import LlmResponse
from google.adk.utils.context_utils import Aclosing
from google.genai import Client, types
from pydantic import BaseModel, Field
from typing_extensions import override

# Session keys for the user's creative brief (set by BookMakerHostAgent before the pipeline runs).
STATE_USER_THEME = "story_user_theme"
STATE_USER_BRIEF = "story_user_brief"

_INTAKE_PROMPT = """Welcome to the Story Book Maker.

I'll turn your ideas into a **5-page illustrated children's story**.

Please send:

1. **Theme** — what the story is about (e.g. friendship, space, kindness).
2. **Brief story idea** — a short description of what happens.

You can send everything in **one message**, for example:

**Theme:** a curious robot in a garden  
**Brief:** The robot finds a tiny seed and learns to care for it until it blooms.

Or send your **theme** first; I'll ask for your story idea next."""

# Shorter "how to send your brief" — used after `_CAPABILITIES_REPLY` so we do not repeat the welcome / 5-page pitch.
_INTAKE_HOW_TO_FORMAT = """**How to start**

Send:

1. **Theme** — what the story is about (e.g. friendship, space, kindness).
2. **Brief story idea** — what happens, in a few sentences.

**Example (one message):**

**Theme:** a curious robot in a garden  
**Brief:** The robot finds a tiny seed and learns to care for it until it blooms.

Or send your **theme** first; I'll ask for your story idea next."""

_CAPABILITIES_REPLY = """Here is what I can do:

I am a **children's story book maker**. After you send a **theme** and **brief story idea**, I:

- Write a **5-page** children's story (short, age-appropriate text).
- Create a **matching illustration for each page**.

"""

_GUARDRAIL_OUT_OF_SCOPE = (
    "That doesn’t look like **storybook input** for this app. I only help you make a **children’s picture book**: "
    "you share a **theme** (what the fictional story is about) and a **brief story idea** (what happens), "
    "then I write **5 short pages** with **one illustration each**.\n\n"
    "Please send creative input for a **made-up kids’ story**, not general tasks, chat, or unrelated questions.\n\n"
    "**Example:**\n\n"
    "**Theme:** brave mice in a bakery  \n"
    "**Brief:** They team up to save the last cinnamon roll before the shop opens.\n"
)


class StoryPage(BaseModel):
    page_number: int = Field(description="Page number, starting from 1.")
    text: str = Field(description="Short, child-friendly page text.")
    visual: str = Field(description="Visual description for illustration.")


class StoryPlan(BaseModel):
    title: str = Field(description="Title of the children's story.")
    pages: list[StoryPage] = Field(description="Exactly 5 pages for the story.")


def _user_text_from_content(user_content: types.Content | None) -> str:
    if not user_content or not user_content.parts:
        return ""
    return "".join(
        part.text or "" for part in user_content.parts if part.text and not part.thought
    ).strip()


def _parse_labeled_theme_brief(text: str) -> tuple[str | None, str | None]:
    """Parse lines like `Theme: ...` and `Brief:` / `Idea:` / `Story idea:`."""
    if not text.strip():
        return None, None
    theme_m = re.search(
        r"(?is)(?:^|\n)\s*(?:theme|topic)\s*:\s*(.+?)(?=(?:^|\n)\s*(?:brief|story\s*idea|idea)\s*:|$)",
        text,
    )
    brief_m = re.search(r"(?is)(?:^|\n)\s*(?:brief|story\s*idea|idea)\s*:\s*(.+)$", text)
    theme = theme_m.group(1).strip() if theme_m else None
    brief = brief_m.group(1).strip() if brief_m else None
    if theme and not brief:
        tail = text[theme_m.end() :].strip() if theme_m else ""
        if tail and not re.match(r"(?is)^\s*(?:brief|story\s*idea|idea)\s*:", tail):
            brief = tail
    return theme, brief


def _split_two_blocks(text: str) -> tuple[str | None, str | None]:
    """Split on a blank line into theme-like and brief-like blocks."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def _is_small_talk(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 2:
        return True
    return bool(
        re.fullmatch(
            r"(hi+|hello+|hey+|yo+|start|go|ok|yes|thanks+|thank you|thx|cool|great|nice|awesome)[\s!.?]*",
            t,
        )
    )


def _is_meta_or_help_question(text: str) -> bool:
    """True if the user is asking what the agent does / how it works, not giving a story theme."""
    t = text.strip().lower()
    if not t:
        return False
    patterns = (
        r"what\s+can\s+you\s+do",
        r"what\s+do\s+you\s+do",
        r"what\s+are\s+you",
        r"who\s+are\s+you",
        r"what\s+can\s+this\s+do",
        r"how\s+does\s+(this|it)\s+work",
        r"how\s+do\s+you\s+work",
        r"what\s+do\s+you\s+offer",
        r"what\s+happens\s+here",
        r"tell\s+me\s+about\s+(yourself|this)",
        r"\bhelp\b(?:\s*!)?\s*$",
        r"\b(capabilities|features)\b.*\b(you|this)\b",
        r"\bhow\s+to\s+use\b",
        r"what\s+is\s+this",
        r"what\s+are\s+you\s+for",
    )
    return any(re.search(p, t) for p in patterns)


def _parse_intake_gate_json(raw: str) -> bool | None:
    """Parse model JSON `{\"allowed\": true|false}`. Returns None if invalid."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "allowed" in obj:
            return bool(obj["allowed"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _build_intake_gate_prompt(
    user_text: str,
    mode: Literal["theme_or_full", "brief_only"],
    saved_theme: str,
) -> str:
    rules = """You validate messages for a children's PICTURE-BOOK app.

The app ONLY accepts creative input for a fictional children's story:
1) **Theme** — what the story is about (topic, mood, world, age-appropriate subject).
2) **Brief story idea** — what happens in that story (characters, problem, events). It must be about a made-up kids' tale, not real-world errands or services.

ALLOW (allowed: true) ONLY when the user is clearly supplying or refining content for inventing a children's story (theme and/or plot).

REJECT (allowed: false) for ANYTHING else, including: shopping, chores, homework, coding, email, calendars, travel booking, medical/legal/financial advice, news, politics, general chat, insults, unrelated Q&A, requests for the assistant that are not story creation, or text that is not about a fictional children's book.

Short greetings alone are not handled here. If a message mixes a small hello with a real theme/brief, ALLOW when the story part is clear."""

    if mode == "brief_only":
        return (
            f"{rules}\n\n"
            f"A **theme** is already saved (context only): {saved_theme!r}\n\n"
            "The user's NEW message must be ONLY a **brief fictional story idea** (what happens in the book). "
            "If it is an unrelated request or not a plot description, REJECT.\n\n"
            f'User message:\n"""\n{user_text}\n"""\n\n'
            'Reply with JSON only, no markdown: {"allowed": true} or {"allowed": false}'
        )

    return (
        f"{rules}\n\n"
        f'User message (may include Theme:/Brief: labels or freeform):\n"""\n{user_text}\n"""\n\n'
        'Reply with JSON only, no markdown: {"allowed": true} or {"allowed": false}'
    )


def _sync_llm_intake_gate_allow(
    user_text: str,
    mode: Literal["theme_or_full", "brief_only"],
    saved_theme: str,
) -> bool:
    """Classifier: True only if message is on-topic for storybook intake.

    Fails closed on model errors. If ``GOOGLE_API_KEY`` is unset, the gate is skipped
    (returns True) so local runs without Gemini are not blocked.
    """
    if not user_text.strip():
        return False
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        return True
    model = os.getenv("BOOK_MAKER_INTAKE_GATE_MODEL", "gemini-2.5-flash").strip()
    prompt = _build_intake_gate_prompt(user_text, mode, saved_theme)
    try:
        client = Client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
    except Exception:
        return False

    raw_parts: list[str] = []
    for cand in response.candidates or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in content.parts or []:
            if part.text and not part.thought:
                raw_parts.append(part.text)
    raw = "".join(raw_parts).strip()
    parsed = _parse_intake_gate_json(raw)
    return parsed is True


async def _llm_intake_gate_allow(
    user_text: str,
    *,
    mode: Literal["theme_or_full", "brief_only"],
    saved_theme: str = "",
) -> bool:
    return await asyncio.to_thread(_sync_llm_intake_gate_allow, user_text, mode, saved_theme)


async def _apply_intake_from_message(tc: Context, user_text: str) -> tuple[bool, str]:
    """Update session state from the user's message.

    Returns:
        (ready_to_run_book_pipeline, reply_text_if_not_ready).
        When not ready, reply_text is shown to the user (intake continues).
    """
    theme_in = (tc.state.get(STATE_USER_THEME) or "").strip()
    brief_in = (tc.state.get(STATE_USER_BRIEF) or "").strip()

    if theme_in and brief_in:
        lt, lb = _parse_labeled_theme_brief(user_text)
        if lt and lb:
            if not await _llm_intake_gate_allow(user_text, mode="theme_or_full", saved_theme=""):
                return False, _GUARDRAIL_OUT_OF_SCOPE
            tc.state[STATE_USER_THEME] = lt
            tc.state[STATE_USER_BRIEF] = lb
            return True, ""
        if _is_small_talk(user_text):
            return (
                False,
                "When you're ready for another book, send your **theme** and **brief story idea** again.",
            )
        return True, ""

    lt, lb = _parse_labeled_theme_brief(user_text)
    if lt and lb:
        if not await _llm_intake_gate_allow(user_text, mode="theme_or_full", saved_theme=""):
            return False, _GUARDRAIL_OUT_OF_SCOPE
        tc.state[STATE_USER_THEME] = lt
        tc.state[STATE_USER_BRIEF] = lb
        return True, ""

    if lt and not lb:
        if not await _llm_intake_gate_allow(user_text, mode="theme_or_full", saved_theme=""):
            return False, _GUARDRAIL_OUT_OF_SCOPE
        tc.state[STATE_USER_THEME] = lt
        preview = lt[:90] + ("…" if len(lt) > 90 else "")
        return (
            False,
            f"I've noted your **theme**: {preview}\n\n"
            "Now please send your **brief story idea** — what happens in the story?",
        )

    bt, bb = _split_two_blocks(user_text)
    if bt and bb and len(user_text) >= 20:
        if not await _llm_intake_gate_allow(user_text, mode="theme_or_full", saved_theme=""):
            return False, _GUARDRAIL_OUT_OF_SCOPE
        tc.state[STATE_USER_THEME] = bt
        tc.state[STATE_USER_BRIEF] = bb
        return True, ""

    if theme_in and not brief_in:
        if _is_meta_or_help_question(user_text):
            if _is_meta_or_help_question(theme_in):
                tc.state[STATE_USER_THEME] = ""
                tc.state[STATE_USER_BRIEF] = ""
                return False, _CAPABILITIES_REPLY + _INTAKE_HOW_TO_FORMAT
            preview = theme_in[:90] + ("…" if len(theme_in) > 90 else "")
            return (
                False,
                _CAPABILITIES_REPLY
                + f"I already have this **theme** saved: {preview}\n\n"
                "Send your **brief story idea** next — what happens in the story?",
            )
        if not await _llm_intake_gate_allow(
            user_text, mode="brief_only", saved_theme=theme_in
        ):
            return (
                False,
                _GUARDRAIL_OUT_OF_SCOPE
                + "\n\nPlease send a **brief story idea** for the theme you already chose — "
                "what happens in the book? (A few sentences about characters and events.)",
            )
        tc.state[STATE_USER_BRIEF] = user_text.strip()
        return True, ""

    if not theme_in:
        if _is_meta_or_help_question(user_text):
            return False, _CAPABILITIES_REPLY + _INTAKE_HOW_TO_FORMAT
        if _is_small_talk(user_text):
            return False, _INTAKE_PROMPT
        lines = [ln.strip() for ln in user_text.split("\n") if ln.strip()]
        if len(lines) >= 2 and len(user_text) >= 30:
            if not await _llm_intake_gate_allow(user_text, mode="theme_or_full", saved_theme=""):
                return False, _GUARDRAIL_OUT_OF_SCOPE
            tc.state[STATE_USER_THEME] = lines[0]
            tc.state[STATE_USER_BRIEF] = "\n".join(lines[1:])
            return True, ""
        if user_text.strip():
            if not await _llm_intake_gate_allow(user_text, mode="theme_or_full", saved_theme=""):
                return False, _GUARDRAIL_OUT_OF_SCOPE
            tc.state[STATE_USER_THEME] = user_text.strip()
            preview = user_text.strip()
            if len(preview) > 90:
                preview = preview[:87] + "…"
            return (
                False,
                f"I've saved your **theme**: {preview}\n\n"
                "Now please send your **brief story idea** — what happens in the story? "
                "(A few sentences is enough.)",
            )
        return False, _INTAKE_PROMPT

    return False, _INTAKE_PROMPT


async def _story_writer_instruction(ctx: ReadonlyContext) -> str:
    theme = (ctx.state.get(STATE_USER_THEME) or "").strip() or "(not provided)"
    brief = (ctx.state.get(STATE_USER_BRIEF) or "").strip() or "(not provided)"
    return (
        "You are a children's story writer.\n"
        f"The reader chose this **theme**: {theme}\n"
        f"Their **brief story idea** (what should happen): {brief}\n\n"
        "Produce a story plan with EXACTLY 5 pages based only on that theme and brief.\n"
        "Each page must include:\n"
        "- page_number (1..5)\n"
        "- text (1-2 short child-friendly sentences)\n"
        "- visual (a concise illustration prompt)\n"
        "Keep tone positive and age-appropriate."
    )


def _generate_openai_image_bytes(prompt: str, image_model: str) -> tuple[bytes | None, str | None]:
    """Generate image bytes from OpenAI Images API."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "OPENAI_API_KEY is missing."

    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": image_model,
            "prompt": prompt,
            "size": "1024x1024",
        },
        timeout=120,
    )
    if response.status_code >= 400:
        return None, f"OpenAI image API error ({response.status_code}): {response.text[:400]}"

    payload = response.json()
    items = payload.get("data", [])
    if not items:
        return None, "OpenAI image API returned no data."

    b64_image = items[0].get("b64_json")
    if not b64_image:
        return None, "OpenAI response did not include `b64_json`."

    return b64decode(b64_image), None


async def _generate_one_page_artifact(
    tool_context: Context,
    page_number: int,
    visual_prompt: str,
) -> dict[str, Any]:
    """Create one image artifact for a story page; merge deltas into `tool_context.actions`."""

    image_provider = os.getenv("BOOK_MAKER_IMAGE_PROVIDER", "openai").strip().lower()
    image_model = os.getenv(
        "BOOK_MAKER_IMAGE_MODEL",
        "gpt-image-1" if image_provider == "openai" else "gemini-2.5-flash-image-preview",
    )
    client = Client() if image_provider == "google" else None
    prompt = (
        "Create a colorful, child-friendly storybook illustration. "
        "Soft shapes, warm light, no text on image. "
        f"Scene: {visual_prompt}"
    )

    model_error: str | None = None
    image_bytes: bytes | None = None
    try:
        if image_provider == "openai":
            image_bytes, model_error = _generate_openai_image_bytes(prompt, image_model)
            response = None
        else:
            response = client.models.generate_content(
                model=image_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
    except Exception as exc:  # noqa: BLE001
        response = None
        model_error = str(exc)

    image_saved = False
    record: dict[str, Any] = {
        "page_number": page_number,
        "visual": visual_prompt,
        "image_provider": image_provider,
        "image_model": image_model,
    }

    if image_provider == "openai" and image_bytes:
        filename = f"page_{page_number}.png"
        artifact = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )
        version = await tool_context.save_artifact(filename, artifact)
        record.update(
            {
                "filename": filename,
                "artifact_version": version,
            }
        )
        image_saved = True
    elif response is not None:
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if part.inline_data and part.inline_data.data:
                    filename = f"page_{page_number}.png"
                    artifact = types.Part.from_bytes(
                        data=part.inline_data.data,
                        mime_type=part.inline_data.mime_type or "image/png",
                    )
                    version = await tool_context.save_artifact(filename, artifact)
                    record.update(
                        {
                            "filename": filename,
                            "artifact_version": version,
                        }
                    )
                    image_saved = True
                    break
            if image_saved:
                break

    if not image_saved:
        filename = f"page_{page_number}.txt"
        fallback_text = (
            f"Image was not returned by model.\n"
            f"Model: {image_model}\n"
            f"Page: {page_number}\n"
            f"Prompt: {prompt}\n"
            f"Error: {model_error or 'No image bytes returned in response.'}"
        )
        artifact = types.Part.from_text(text=fallback_text)
        version = await tool_context.save_artifact(filename, artifact)
        record.update(
            {
                "filename": filename,
                "artifact_version": version,
                "note": "Fallback text artifact (model unavailable or no image bytes).",
                "error": model_error,
            }
        )

    return record


def _on_writer_start(callback_context: CallbackContext) -> None:
    callback_context.state["pipeline_progress"] = "Creating your story..."
    return None


def _writer_after_model(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Keep structured `story_plan` in state but show a short story in the chat (not raw JSON)."""
    if not llm_response.content or not llm_response.content.parts:
        return None
    raw = "".join(
        part.text or ""
        for part in llm_response.content.parts
        if part.text and not part.thought
    )
    if not raw.strip():
        return None
    try:
        plan = StoryPlan.model_validate_json(raw)
    except Exception:
        try:
            plan = StoryPlan.model_validate(json.loads(raw))
        except Exception:
            return None
    callback_context.state["story_plan"] = plan.model_dump()
    lines: list[str] = [f"**{plan.title}**", ""]
    for pg in plan.pages:
        lines.append(f"**Page {pg.page_number}** — {pg.text}")
    friendly = "\n".join(lines)
    return llm_response.model_copy(
        update={
            "content": types.Content(
                role="model",
                parts=[types.Part(text=friendly)],
            )
        }
    )


def _make_before_page_image(page_index: int):
    def _cb(callback_context: CallbackContext) -> None:
        callback_context.state["pipeline_progress"] = f"Generating image {page_index}/5"
        return None

    return _cb


def _on_assembler_start(callback_context: CallbackContext) -> None:
    callback_context.state["pipeline_progress"] = "Assembling your storybook (text + images)..."
    return None


class IllustratePageAgent(BaseAgent):
    """Generates one page illustration; runs inside ParallelAgent."""

    page_index: int = Field(ge=1, le=5, description="1-based page index for this branch.")

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        tc = Context(ctx)
        story_plan = tc.state.get("story_plan")
        if not story_plan:
            tc.state[f"illustration_page_{self.page_index}"] = {
                "page_number": self.page_index,
                "error": "No `story_plan` in state.",
            }
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=f"Page {self.page_index}: missing story plan; skipped illustration."
                        )
                    ],
                ),
                actions=tc.actions,
            )
            return

        pages = story_plan.get("pages", [])
        page_data = next((p for p in pages if p.get("page_number") == self.page_index), None)
        if not page_data:
            tc.state[f"illustration_page_{self.page_index}"] = {
                "page_number": self.page_index,
                "error": "No matching page in story_plan.pages.",
            }
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=f"Page {self.page_index}: page not found in plan; skipped."
                        )
                    ],
                ),
                actions=tc.actions,
            )
            return

        visual = page_data.get("visual", "")
        result = await _generate_one_page_artifact(tc, self.page_index, visual)
        tc.state[f"illustration_page_{self.page_index}"] = result

        fn = result.get("filename", "?")
        status = result.get("note") or "saved to Artifacts"
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"Page {self.page_index}/5 illustration `{fn}` — {status}."
                        )
                    )
                ],
            ),
            actions=tc.actions,
        )


class AssembleBookAgent(BaseAgent):
    """Merges parallel page results, sets `illustrations` and emits the final storybook text."""

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        tc = Context(ctx)
        story_plan = tc.state.get("story_plan")
        if not story_plan:
            tc.state["pipeline_progress"] = "Book assembly failed: no story_plan."
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Could not assemble book: missing story plan.")],
                ),
                actions=tc.actions,
            )
            return

        illustrations: list[dict[str, Any]] = []
        for i in range(1, 6):
            key = f"illustration_page_{i}"
            entry = tc.state.get(key)
            if isinstance(entry, dict):
                illustrations.append(entry)

        illustrations.sort(key=lambda x: x.get("page_number", 0))
        tc.state["illustrations"] = illustrations

        title = story_plan.get("title", "Untitled")
        header = (
            f"# {title}\n\n"
            "Below is your **5-page storybook**. Each **following message** is one page: "
            "story text plus its illustration (so every image shows in the chat).\n"
        )
        by_page = {p.get("page_number"): p for p in story_plan.get("pages", [])}

        book_markdown = "\n\n".join(
            [header]
            + [
                f"## Page {ill.get('page_number')}\n**Story text:**\n"
                f"{by_page.get(ill.get('page_number'), {}).get('text', '')}\n**Image:** `{ill.get('filename')}`"
                for ill in illustrations
            ]
        ).strip()
        tc.state["story_book_markdown"] = book_markdown
        tc.state["pipeline_progress"] = "Done — your storybook is ready."
        tc.state[STATE_USER_THEME] = ""
        tc.state[STATE_USER_BRIEF] = ""

        # One Event per page: many UIs only render the last image if several image
        # parts are bundled in a single message. Separate events preserve each illustration.
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=header)]),
            actions=EventActions(),
        )

        n_pages = len(illustrations)
        for idx, ill in enumerate(illustrations):
            pn = ill.get("page_number")
            page_text = by_page.get(pn, {}).get("text", "") if by_page else ""
            fn = ill.get("filename", "")
            parts_page: list[types.Part] = [
                types.Part(text=f"## Page {pn}\n\n**Story text:**\n{page_text}\n"),
            ]
            if fn.endswith(".png"):
                try:
                    loaded = await tc.load_artifact(fn)
                except Exception:
                    loaded = None
                if loaded and getattr(loaded, "inline_data", None):
                    idata = loaded.inline_data
                    parts_page.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=idata.mime_type or "image/png",
                                data=idata.data,
                            )
                        )
                    )
                elif loaded and getattr(loaded, "file_data", None):
                    parts_page.append(loaded)
                else:
                    parts_page.append(
                        types.Part(
                            text=f"*(Illustration `{fn}` — open it from the Artifacts panel.)*\n"
                        )
                    )
            elif fn.endswith(".txt"):
                parts_page.append(
                    types.Part(
                        text=f"*(Page {pn} image unavailable; see artifact `{fn}` for details.)*\n"
                    )
                )

            is_last = idx == n_pages - 1
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(role="model", parts=parts_page),
                actions=tc.actions if is_last else EventActions(),
            )


def _page_agents() -> list[IllustratePageAgent]:
    return [
        IllustratePageAgent(
            name=f"illustrate_page_{i}_agent",
            description=f"Generates the illustration for story page {i} of 5.",
            page_index=i,
            before_agent_callback=_make_before_page_image(i),
        )
        for i in range(1, 6)
    ]


story_writer_agent = LlmAgent(
    name="story_writer_agent",
    model="gemini-2.5-flash",
    description="Writes a 5-page children's story plan from the user's theme and brief in session state.",
    instruction=_story_writer_instruction,
    output_schema=StoryPlan,
    after_model_callback=_writer_after_model,
    before_agent_callback=_on_writer_start,
)


parallel_image_agent = ParallelAgent(
    name="parallel_image_agent",
    description="Generates all five page illustrations concurrently.",
    sub_agents=_page_agents(),
)


assemble_book_agent = AssembleBookAgent(
    name="assemble_book_agent",
    description="Combines story text and image artifacts into the final storybook output.",
    before_agent_callback=_on_assembler_start,
)


book_build_pipeline = SequentialAgent(
    name="book_build_pipeline",
    description=(
        "Writes the story, generates five page illustrations in parallel, then assembles output."
    ),
    sub_agents=[story_writer_agent, parallel_image_agent, assemble_book_agent],
)


class BookMakerHostAgent(BaseAgent):
    """Collects theme + brief from the user, then runs the illustration pipeline."""

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        tc = Context(ctx)
        user_text = _user_text_from_content(ctx.user_content)
        ready, reply = await _apply_intake_from_message(tc, user_text)
        if not ready:
            tc.state["pipeline_progress"] = "Waiting for your theme and brief story idea."
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(role="model", parts=[types.Part(text=reply)]),
                actions=tc.actions,
            )
            return

        async with Aclosing(book_build_pipeline.run_async(ctx)) as agen:
            async for event in agen:
                yield event


root_agent = BookMakerHostAgent(
    name="book_maker_root_agent",
    description=(
        "Asks for a theme and brief story idea, then builds a 5-page illustrated storybook."
    ),
)


def pretty_story_from_state(state: dict[str, Any]) -> str:
    """Helper for local debugging to format story plan text."""

    story_plan = state.get("story_plan")
    illustrations = state.get("illustrations", [])
    book_md = state.get("story_book_markdown")
    if book_md:
        return book_md
    if not story_plan:
        return "No story plan found in state."

    lines: list[str] = [f"Title: {story_plan['title']}"]
    illustration_map = {item["page_number"]: item.get("filename") for item in illustrations}
    for page in story_plan["pages"]:
        lines.append("")
        lines.append(f"Page {page['page_number']}:")
        lines.append(f'Text: "{page["text"]}"')
        lines.append(f'Visual: "{page["visual"]}"')
        image_name = illustration_map.get(page["page_number"], "(not generated yet)")
        lines.append(f"Image Artifact: {image_name}")
    return "\n".join(lines)


def story_plan_as_json(state: dict[str, Any]) -> str:
    """Helper for local debugging to print story plan JSON."""

    return json.dumps(state.get("story_plan", {}), ensure_ascii=True, indent=2)

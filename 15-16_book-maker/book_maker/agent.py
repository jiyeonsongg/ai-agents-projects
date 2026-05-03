"""Google ADK multi-agent app for a children's story book maker."""

from __future__ import annotations

import json
import os
from base64 import b64decode
from typing import Any, AsyncGenerator

import requests
from google.adk.agents import BaseAgent, Context, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_response import LlmResponse
from google.genai import Client, types
from pydantic import BaseModel, Field
from typing_extensions import override


class StoryPage(BaseModel):
    page_number: int = Field(description="Page number, starting from 1.")
    text: str = Field(description="Short, child-friendly page text.")
    visual: str = Field(description="Visual description for illustration.")


class StoryPlan(BaseModel):
    title: str = Field(description="Title of the children's story.")
    pages: list[StoryPage] = Field(description="Exactly 5 pages for the story.")


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

        lines: list[str] = [
            f"# {story_plan.get('title', 'Untitled')}",
            "",
            "Here is your storybook: each section has the page text and the matching image artifact.",
            "",
        ]
        by_page = {p.get("page_number"): p for p in story_plan.get("pages", [])}
        for ill in illustrations:
            pn = ill.get("page_number")
            page_text = by_page.get(pn, {}).get("text", "") if by_page else ""
            fn = ill.get("filename", "(none)")
            lines.append(f"## Page {pn}")
            lines.append("")
            lines.append(f"**Text:** {page_text}")
            lines.append("")
            lines.append(f"**Image artifact:** `{fn}` — open it from the Artifacts panel.")
            lines.append("")

        book_markdown = "\n".join(lines).strip()
        tc.state["story_book_markdown"] = book_markdown
        tc.state["pipeline_progress"] = "Done — your storybook is ready."

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part(text=book_markdown)],
            ),
            actions=tc.actions,
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
    description="Writes a 5-page children's story plan from a user theme.",
    instruction=(
        "You are a children's story writer.\n"
        "Read the user's theme and produce a story plan with EXACTLY 5 pages.\n"
        "Each page must include:\n"
        "- page_number (1..5)\n"
        "- text (1-2 short child-friendly sentences)\n"
        "- visual (a concise illustration prompt)\n"
        "Keep tone positive and age-appropriate."
    ),
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


root_agent = SequentialAgent(
    name="book_maker_root_agent",
    description=(
        "Sequential pipeline: writer, parallel illustrators (5 images), then book assembly."
    ),
    sub_agents=[story_writer_agent, parallel_image_agent, assemble_book_agent],
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

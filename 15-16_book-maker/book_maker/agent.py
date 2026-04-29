"""Google ADK multi-agent app for a children's story book maker."""

from __future__ import annotations

import json
import os
from base64 import b64decode
from typing import Any

import requests
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import ToolContext
from google.genai import Client, types
from pydantic import BaseModel, Field


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


async def illustrate_story_from_state(tool_context: ToolContext) -> dict[str, Any]:
    """Generate one image artifact per page using the story plan in state."""

    story_plan = tool_context.state.get("story_plan")
    if not story_plan:
        return {"status": "error", "message": "No `story_plan` found in agent state."}

    pages = story_plan.get("pages", [])
    if not pages:
        return {"status": "error", "message": "`story_plan.pages` is empty."}

    image_provider = os.getenv("BOOK_MAKER_IMAGE_PROVIDER", "openai").strip().lower()
    image_model = os.getenv(
        "BOOK_MAKER_IMAGE_MODEL",
        "gpt-image-1" if image_provider == "openai" else "gemini-2.5-flash-image-preview",
    )
    client = Client() if image_provider == "google" else None
    artifacts: list[dict[str, Any]] = []

    for page in pages:
        page_number = page["page_number"]
        visual_prompt = page["visual"]
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
            # Keep the workflow alive even when image model access is restricted.
            response = None
            model_error = str(exc)

        image_saved = False
        if image_provider == "openai" and image_bytes:
            filename = f"page_{page_number}.png"
            artifact = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            )
            version = await tool_context.save_artifact(filename, artifact)
            artifacts.append(
                {
                    "page_number": page_number,
                    "filename": filename,
                    "artifact_version": version,
                    "visual": visual_prompt,
                    "image_provider": image_provider,
                    "image_model": image_model,
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
                        artifacts.append(
                            {
                                "page_number": page_number,
                                "filename": filename,
                                "artifact_version": version,
                                "visual": visual_prompt,
                                "image_provider": image_provider,
                                "image_model": image_model,
                            }
                        )
                        image_saved = True
                        break
                if image_saved:
                    break

        if not image_saved:
            # Fallback artifact so each page still has output in Artifact panel.
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
            artifacts.append(
                {
                    "page_number": page_number,
                    "filename": filename,
                    "artifact_version": version,
                    "visual": visual_prompt,
                    "image_provider": image_provider,
                    "image_model": image_model,
                    "note": "Fallback text artifact (model unavailable or no image bytes).",
                    "error": model_error,
                }
            )

    tool_context.state["illustrations"] = artifacts
    return {
        "status": "ok",
        "pages_processed": len(pages),
        "artifacts": artifacts,
    }


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
    output_key="story_plan",
)


illustrator_agent = LlmAgent(
    name="illustrator_agent",
    model="gemini-2.5-flash",
    description=(
        "Reads story plan from state and generates one illustration artifact per page."
    ),
    instruction=(
        "You are the story illustrator.\n"
        "Call the `illustrate_story_from_state` tool to generate image artifacts from "
        "the story plan stored in state. Then report concise results."
    ),
    tools=[illustrate_story_from_state],
)


root_agent = SequentialAgent(
    name="book_maker_root_agent",
    description="Creates a 5-page children's story and illustration artifacts.",
    sub_agents=[story_writer_agent, illustrator_agent],
)


def pretty_story_from_state(state: dict[str, Any]) -> str:
    """Helper for local debugging to format story plan text."""

    story_plan = state.get("story_plan")
    illustrations = state.get("illustrations", [])
    if not story_plan:
        return "No story plan found in state."

    lines: list[str] = [f"Title: {story_plan['title']}"]
    illustration_map = {item["page_number"]: item["filename"] for item in illustrations}
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


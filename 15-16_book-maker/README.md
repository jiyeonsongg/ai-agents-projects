# Children's Story Book Maker (Google ADK)

This project defines a two-agent workflow:

- `story_writer_agent`: receives the user's theme and writes a 5-page story plan (`text` + `visual` per page) into Agent State (`story_plan`).
- `illustrator_agent`: reads `story_plan` from state and generates one image artifact per page (`page_1.png` ... `page_5.png`).

## Requirements

- Python 3.13+
- A Google Generative AI key in `.env`:
  - `GOOGLE_API_KEY=your_key_here`

The app auto-loads `.env` when `book_maker` is imported, so `adk web` can use
the key without manually exporting it in your shell.

Image generation defaults to OpenAI. Add these to `.env`:

- `OPENAI_API_KEY=your_openai_api_key`
- `BOOK_MAKER_IMAGE_PROVIDER=openai`
- `BOOK_MAKER_IMAGE_MODEL=gpt-image-1`

If you want Google image generation instead:

- `BOOK_MAKER_IMAGE_PROVIDER=google`
- `BOOK_MAKER_IMAGE_MODEL=gemini-2.5-flash-image-preview`

## Run with ADK Web UI

From this project folder:

```bash
uv sync
uv run adk web .
```

In the ADK Web UI:

1. Select app: `book_maker`
2. Send a prompt/theme such as:
   - `Create a story about a brave dragon helping a lost princess`

The agents will:

1. Build 5 pages of story content in state.
2. Generate and save per-page image artifacts.

You should see output similar to:

- Page 1 text and visual description
- Page 1 image artifact
- ...
- Page 5 image artifact


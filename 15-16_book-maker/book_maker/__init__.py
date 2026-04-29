"""Children's story book maker ADK app."""

from pathlib import Path

from dotenv import load_dotenv

# Load project .env so ADK Web UI can access GOOGLE_API_KEY automatically.
# override=True ensures a blank shell env var does not mask the .env value.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from .agent import root_agent


"""The simulated user: a cheap model holding one task's user script.

`reply` answers the conversation so far as the user would: it answers the
client's questions from the script, approves or turns down a plan card by the
script's rule, and says `done` once the client has answered or stopped. The
script stays with this model; the client sees only its replies.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel
from pydantic_ai import Agent
from tasks import load_user

USER_MODEL = "google:gemini-3.5-flash-lite"

USER_INSTRUCTION = """\
You play a user who asked an assistant a question about development finance data.
Your script says what you mean and how you answer. Reply as that user, in one or two
plain sentences:

- When the assistant asks you something, answer as the script says. Pick from the
  options the assistant offers where one matches the script.
- When the assistant shows a plan and asks for approval, apply the script's
  approval rule.
- Set `done` when the assistant has given its final answer with figures, has said
  it cannot answer, or keeps going without needing you. Leave `reply` empty then.

Never mention the script. Never ask for anything the script does not hold.
"""


class UserTurn(BaseModel):
    reply: str
    done: bool


def opening(task_id: str) -> str:
    """What the user says first: the task's question."""
    return load_user(task_id)["opening"]


def transcript(turns: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(f"{who}: {text}" for who, text in turns)


async def reply(task_id: str, turns: Sequence[tuple[str, str]]) -> UserTurn:
    """The user's next turn after `turns`, `("User" | "Assistant", text)` pairs.

    The model client retries a failed request three times, so a connect timeout
    to the model does not end a conversation.
    """
    user = Agent(
        USER_MODEL,
        instructions=USER_INSTRUCTION + "\nScript:\n" + json.dumps(load_user(task_id), indent=1),
        output_type=UserTurn,
        model_settings={"max_retries": 3},
    )
    return (await user.run(transcript(turns))).output

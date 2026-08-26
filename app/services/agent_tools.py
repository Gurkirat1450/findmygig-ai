"""
The tool-calling loop, spelled out manually so you can see every step —
this is what frameworks like LangGraph's prebuilt agents do for you
automatically, once you get to Day 3.

The loop, in plain terms:
1. Send the user's message + tool descriptions to the LLM.
2. The LLM decides: answer directly, OR call a tool (it never runs the
   tool itself — it just says "call filter_gigs_by_skill with skill=Docker").
3. WE execute the actual tool function (the LLM has no code execution of
   its own) and get a real result back.
4. We send that result back to the LLM as a new message.
5. The LLM now has real data and generates its final answer.

This is exactly what "agentic AI vs. a single LLM call" means in practice —
a single call can only use what's in its training data or the prompt; a
tool-calling loop lets it fetch fresh, exact data mid-conversation.
"""

import os

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.services.tools import ALL_TOOLS

_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def chat_with_tools(user_message: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = [HumanMessage(content=user_message)]

    # Step 1-2: ask the LLM, see if it wants to call a tool.
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # A model can request multiple tool calls in one turn — handle all of them.
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_fn = _TOOLS_BY_NAME[tool_call["name"]]
            # Step 3: WE actually run the tool — the LLM only requested it.
            tool_result = tool_fn.invoke(tool_call["args"])
            # Step 4: feed the real result back as a ToolMessage.
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

        # Step 5: call the LLM again, now with real tool results in context,
        # so it can generate a final answer grounded in that data.
        final_response = llm_with_tools.invoke(messages)
        # .text (not .content) — current langchain-google-genai returns content
        # as a list of structured blocks (carrying the thought signature
        # alongside the text) rather than a plain string; .text extracts
        # just the text either way.
        return final_response.text

    # No tool was needed — the model answered directly.
    return response.text

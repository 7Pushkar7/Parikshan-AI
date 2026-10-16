"""The agent itself: a ReAct loop built as a LangGraph state machine.

           +-----------+   has tool calls?   +-----------+
  START -> |   agent   | ------- yes ------> |   tools   |
           | (Gemini)  | <------------------ | (Python)  |
           +-----------+                     +-----------+
                 | no tool calls (final answer)
                 v
                END

- "agent" node : one LLM call. Gemini reads the chat + tool results and either asks for
                 tool calls (ACT) or writes the final answer.
- "tools" node : runs the requested Python functions and adds their results as ToolMessages (OBSERVE).
- checkpointer : after every step the full state is saved to SQLite (data/memory.sqlite),
                 keyed by thread_id. That is the agent's short-term (conversation) memory.

Why LangGraph and not the older LangChain `AgentExecutor`? AgentExecutor is the legacy API
(moved to `langchain-classic`). LangGraph is LangChain's current agent runtime: same ReAct
idea, but the loop is an explicit graph, with built-in persistence, streaming and step limits.
"""
from __future__ import annotations

from datetime import date

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.config import get_settings
from app.llm import with_fallback
from app.tools import ALL_TOOLS


def trim_history(messages: list[AnyMessage], max_messages: int) -> list[AnyMessage]:
    """Short-term memory window: keep the most recent messages, always starting at a user
    message so tool calls and their results stay paired. The current turn is always kept."""
    if len(messages) <= max_messages:
        return messages
    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not human_idx:
        return messages
    start = human_idx[-1]
    for i in reversed(human_idx):
        if len(messages) - i <= max_messages:
            start = i
        else:
            break
    return messages[start:]


def build_graph(checkpointer=None):
    settings = get_settings()
    model_with_tools = with_fallback(lambda m: m.bind_tools(ALL_TOOLS))

    async def agent(state: MessagesState, config: RunnableConfig):
        history = trim_history(state["messages"], settings.history_messages_kept)
        system = SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat()))
        response = await model_with_tools.ainvoke([system, *history], config)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)

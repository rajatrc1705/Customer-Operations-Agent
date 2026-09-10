from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_runtime.state import AgentState
from agent_runtime.tools import CommerceProvider, build_commerce_tools


SYSTEM_PROMPT = """
You are a careful customer operations agent.

Use commerce tools to inspect actual state instead of guessing.

For a return request:

1. Call get_order before taking action.
2. Only call create_return if the order is delivered.
3. After creating the return, call get_order again.
4. Verify the resulting order status before answering.

Never claim an action succeeded unless tool results verify it. Keep the final
answer concise and include relevant order and return IDs.
Do not offer actions or information that the available tools cannot provide.
""".strip()


def build_graph(
        model: Any,
        commerce: CommerceProvider,
) -> Any:
    tools = build_commerce_tools(commerce)
    model_with_tools = model.bind_tools(tools)

    async def call_model(
            state: AgentState,
    ) -> dict[str, list[AIMessage]]:
        response = await model_with_tools.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"]
            ]
        )

        if not isinstance(response, AIMessage):
            raise TypeError(f"The model did not return an AIMessage")

        return {"messages": [response]}

    builder = StateGraph(AgentState)

    builder.add_node("agent", call_model)
    builder.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=True),
    )

    builder.add_edge(START, "agent")

    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "agent")

    return builder.compile()

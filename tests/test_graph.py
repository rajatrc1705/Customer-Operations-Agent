import asyncio

from langchain_core.messages import HumanMessage, ToolMessage

from agent_runtime.graph import build_graph
from agent_runtime.tools import LocalCommerceBackend
from tests.fakes import scripted_model_factory


def test_graph_executes_read_write_verify_path() -> None:
    backend = LocalCommerceBackend()
    graph = build_graph(scripted_model_factory(), backend)

    result = asyncio.run(
        graph.ainvoke(
            {
                "case_id": "case-graph-test",
                "messages": [
                    HumanMessage(
                        content=(
                            "Return order ord-1001 because it arrived too late."
                        )
                    )
                ],
            }
        )
    )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert [message.name for message in tool_messages] == [
        "get_order",
        "create_return",
        "get_order",
    ]
    assert backend.get_order("ord-1001").status == "return_approved"
    assert "ret-1001" in str(result["messages"][-1].text)


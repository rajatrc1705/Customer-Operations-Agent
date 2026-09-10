from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


class ScriptedReturnModel:
    def bind_tools(
        self,
        tools: list[Any],
    ) -> "ScriptedReturnModel":
        self.tool_names = {
            tool.name for tool in tools
        }

        return self

    async def ainvoke(
        self,
        messages: list[Any],
    ) -> AIMessage:
        tool_messages = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
        ]

        if not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order",
                        "args": {
                            "order_id": "ord-1001"
                        },
                        "id": "call-get-before",
                    }
                ],
            )

        called_tools = [
            message.name
            for message in tool_messages
        ]

        if "create_return" not in called_tools:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_return",
                        "args": {
                            "order_id": "ord-1001",
                            "reason": "arrived too late",
                        },
                        "id": "call-create-return",
                    }
                ],
            )

        if called_tools.count("get_order") == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order",
                        "args": {
                            "order_id": "ord-1001"
                        },
                        "id": "call-get-after",
                    }
                ],
            )

        return AIMessage(
            content=(
                "Return ret-1001 was approved for "
                "order ord-1001. I verified that the "
                "order is now return_approved."
            )
        )


def scripted_model_factory() -> ScriptedReturnModel:
    return ScriptedReturnModel()
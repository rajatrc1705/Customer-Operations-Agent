from collections.abc import Callable
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agent_runtime.graph import build_graph
from agent_runtime.model import create_model
from agent_runtime.tools import (
    CommerceProvider,
    LocalCommerceBackend,
)


class CaseCreate(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4_000,
    )


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, object]
    call_id: str | None = None


class TrajectoryEntry(BaseModel):
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list
    )


class CaseResponse(BaseModel):
    case_id: str
    status: str
    response: str
    trajectory: list[TrajectoryEntry]


def message_to_trajectory(
    message: Any,
) -> TrajectoryEntry:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = message.type

    tool_calls = [
        ToolCallRecord(
            name=call["name"],
            arguments=call.get("args", {}),
            call_id=call.get("id"),
        )
        for call in getattr(message, "tool_calls", [])
    ]

    return TrajectoryEntry(
        role=role,
        content=str(message.text),
        name=getattr(message, "name", None),
        tool_call_id=getattr(
            message,
            "tool_call_id",
            None,
        ),
        tool_calls=tool_calls,
    )


def create_app(
    *,
    model_factory: Callable[[], Any] = create_model,
    commerce: CommerceProvider | None = None,
    case_id_factory: Callable[[], str] | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Customer Operations Agent Runtime",
        version="0.1.0",
    )

    commerce_backend = commerce or LocalCommerceBackend()
    make_case_id = case_id_factory or (
        lambda: str(uuid4())
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/cases",
        response_model=CaseResponse,
    )
    async def create_case(
        request: CaseCreate,
    ) -> CaseResponse:
        case_id = make_case_id()

        try:
            graph = build_graph(
                model_factory(),
                commerce_backend,
            )

            result = await graph.ainvoke(
                {
                    "case_id": case_id,
                    "messages": [
                        HumanMessage(
                            content=request.message
                        )
                    ],
                }
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=503,
                detail=str(error),
            ) from error

        messages = result["messages"]
        final_message = messages[-1]

        if (
            not isinstance(final_message, AIMessage)
            or final_message.tool_calls
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Agent execution ended without "
                    "a final response"
                ),
            )

        return CaseResponse(
            case_id=case_id,
            status="completed",
            response=str(final_message.text),
            trajectory=[
                message_to_trajectory(message)
                for message in messages
            ],
        )

    return app


app = create_app()
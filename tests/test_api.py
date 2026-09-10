from fastapi.testclient import TestClient

from agent_runtime.api import create_app
from agent_runtime.tools import LocalCommerceBackend
from tests.fakes import scripted_model_factory


def test_create_case_returns_response_and_trajectory() -> None:
    app = create_app(
        model_factory=scripted_model_factory,
        commerce=LocalCommerceBackend(),
        case_id_factory=lambda: "case-api-test",
    )

    with TestClient(app) as client:
        response = client.post(
            "/cases",
            json={
                "message": "Return ord-1001 because it arrived too late."
            },
        )

    assert response.status_code == 200

    body = response.json()
    assert body["case_id"] == "case-api-test"
    assert body["status"] == "completed"
    assert "ret-1001" in body["response"]
    assert [
        entry["name"]
        for entry in body["trajectory"]
        if entry["role"] == "tool"
    ] == [
        "get_order",
        "create_return",
        "get_order",
    ]


def test_health_does_not_require_model_credentials() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_empty_case_message_is_rejected() -> None:
    app = create_app(
        model_factory=scripted_model_factory,
        commerce=LocalCommerceBackend(),
    )

    with TestClient(app) as client:
        response = client.post("/cases", json={"message": ""})

    assert response.status_code == 422


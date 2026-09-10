import pytest

from agent_runtime.tools import LocalCommerceBackend


def test_get_order_returns_typed_order() -> None:
    backend = LocalCommerceBackend()

    order = backend.get_order("ord-1001")

    assert order.order_id == "ord-1001"
    assert order.status == "delivered"
    assert order.item == "Trail running shoes"
    assert order.price == 129.00


def test_unknown_order_is_explicit() -> None:
    backend = LocalCommerceBackend()

    with pytest.raises(ValueError, match="was not found"):
        backend.get_order("ord-missing")


def test_return_changes_order_status() -> None:
    backend = LocalCommerceBackend()

    created_return = backend.create_return(
        "ord-1001",
        "arrived too late",
    )

    assert created_return.return_id == "ret-1001"
    assert created_return.status == "approved"
    assert backend.get_order("ord-1001").status == "return_approved"


def test_processing_order_cannot_be_returned() -> None:
    backend = LocalCommerceBackend()

    with pytest.raises(ValueError, match="cannot be returned"):
        backend.create_return("ord-1002", "changed my mind")


def test_repeated_return_is_safe() -> None:
    backend = LocalCommerceBackend()

    first = backend.create_return("ord-1001", "arrived too late")
    second = backend.create_return("ord-1001", "duplicate request")

    assert second == first


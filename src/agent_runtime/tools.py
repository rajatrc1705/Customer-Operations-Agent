from typing import Protocol

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

class Customer(BaseModel):
    customer_id: str
    name: str
    email: str


class Order(BaseModel):
    order_id: str
    customer_id: str
    status: str
    item: str
    price: float

    currency: str = "USD"

class Return(BaseModel):
    return_id: str
    order_id: str
    status: str
    reason: str


class CommerceProvider(Protocol):
    def get_customer(self, customer_id: str) -> Customer: ...

    def get_order(self, order_id: str) -> Order: ...

    def create_return(self, order_id: str, reason: str) -> Return: ...


def build_commerce_tools(
        provider: CommerceProvider,
) -> list[BaseTool]:
    @tool
    def get_customer(customer_id: str) -> dict[str, object]:
        """Get a customer and their account information by customer ID."""

        return provider.get_customer(customer_id).model_dump()

    @tool
    def get_order(order_id: str) -> dict[str, object]:
        """Get an order and its current status by order ID."""

        return provider.get_order(order_id).model_dump()

    @tool
    def create_return(
            order_id: str,
            reason: str,
    ) -> dict[str, object]:
        """Create a return for a delivered order after inspecting it."""

        return provider.create_return(order_id, reason).model_dump()

    return [get_customer, get_order, create_return]

class LocalCommerceBackend:
    def __init__(self) -> None:
        self._customers = {
            "cust-001": Customer(
                customer_id="cust-001",
                name="Avery Chen",
                email="avery@example.com",
            )
        }

        self._orders = {
            "ord-1001": Order(
                order_id="ord-1001",
                customer_id="cust-001",
                status="delivered",
                item="Trail running shoes",
                price=129.00,
            ),
            "ord-1002": Order(
                order_id="ord-1002",
                customer_id="cust-001",
                status="processing",
                item="Merino hiking socks",
                price=24.00,
            ),
        }

        self._returns: dict[str, Return] = {}

    def get_customer(self, customer_id: str) -> Customer:
        customer = self._customers.get(customer_id)

        if customer is None:
            raise ValueError(f"Customer {customer_id!r} was not found")

        return customer.model_copy(deep=True)

    def get_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id)

        if order is None:
            raise ValueError(f"Order {order_id!r} was not found")

        return order.model_copy(deep=True)

    def create_return(self, order_id: str, reason: str) -> Return:
        return_id = f"ret-{order_id.removeprefix('ord-')}"

        existing_return = self._returns.get(return_id)
        if existing_return is not None:
            return existing_return.model_copy(deep=True)

        order = self.get_order(order_id)

        if order.status != "delivered":
            raise ValueError(
                f"Order {order_id!r} cannot be returned "
                f"from status {order.status!r}"
            )

        created_return = Return(
            return_id=return_id,
            order_id=order_id,
            status="approved",
            reason=reason,
        )

        self._returns[return_id] = created_return
        self._orders[order_id] = order.model_copy(
            update={"status": "return_approved"}
        )

        return created_return.model_copy(deep=True)


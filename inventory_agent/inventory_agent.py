import ast
import json
from smolagents import ToolCallingAgent, OpenAIServerModel
from datetime import datetime
from inventory_agent.inventory_agent_tools import (
    check_item_stock,
    order_item_stock
)

from inventory_agent.inventory_agent_prompts import (
    CHECK_INVENTORY_PROMPT,
    PROCESS_TRANSACTION_PROMPT
)

from project_starter import (
    get_supplier_delivery_date
)

class InventoryAgent(ToolCallingAgent):
    """Agent for managing inventory."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[check_item_stock, order_item_stock],
            model=model,
            name="inventory_agent",
            description="Agent for managing inventory. Check stock levels and order stock items.",
        )

if __name__ == "__main__":
    from project_starter import model
    format_string = "%Y-%m-%d"
    request_list = [
        "tests/quote_agent/test1",
        "tests/quote_agent/test2",
        "tests/quote_agent/test3",
        "tests/quote_agent/test4"
    ]

    for request_path in request_list:
        print(f"Processing request from: {request_path}")
        with open(f"{request_path}/request.json", "r") as f:
            request = json.load(f)

        agent = InventoryAgent(model=model)
        transaction_items = []
        for item in request["items"]:
            delivery_deadline = datetime.strptime(request["delivery_deadline"], format_string)
            result = agent.run(CHECK_INVENTORY_PROMPT.format(item_name=item["item_name"], delivery_deadline=request["delivery_deadline"]))
            item_data = ast.literal_eval(result)
            print(f"item_data: {item_data}")
            item_delivery_deadline_str = get_supplier_delivery_date(request["delivery_deadline"], item["quantity"])
            item_delivery_deadline = datetime.strptime(item_delivery_deadline_str, format_string)
            item_data["item_delivery_deadline"] = item_delivery_deadline
            transaction_item = {
              "item_name": item["item_name"],
              "transaction_type": "sales",
              "quantity": item["quantity"],
              "price": item_data["unit_price"]*item["quantity"],
              "date": item_delivery_deadline_str
            }
            if item_delivery_deadline > delivery_deadline or item_data["current_stock"] < item["quantity"]:
                transaction_item["quantity"] = -1
            transaction_items += [transaction_item]
        quantities = [transaction_item["quantity"] for transaction_item in transaction_items]
        if -1 in quantities:
            print("Could not process quote")
        else:
            # Process quote
            stock_items = []
            for transaction_item in transaction_items:
                agent.run(PROCESS_TRANSACTION_PROMPT.format(transaction_item=json.dumps(transaction_item)))
                result = agent.run(CHECK_INVENTORY_PROMPT.format(item_name=transaction_item["item_name"],
                                                                 delivery_deadline=request["delivery_deadline"]))
                item_data = ast.literal_eval(result)
                stock_balance = item_data["current_stock"] - item_data["min_stock_level"]
                if stock_balance < 0:
                    transaction_item["transaction_type"] = "stock_orders"
                    agent.run(PROCESS_TRANSACTION_PROMPT.format(transaction_item=json.dumps(transaction_item)))


        print("#"*200)

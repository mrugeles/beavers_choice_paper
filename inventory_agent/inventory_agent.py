import ast
import json
from smolagents import ToolCallingAgent, OpenAIServerModel
from inventory_agent.inventory_agent_tools import (
    check_item_stock,
    order_item_stock
)

from inventory_agent.inventory_agent_prompts import (
    CHECK_INVENTORY_PROMPT
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
        for item in request["items"]:
            result = agent.run(CHECK_INVENTORY_PROMPT.format(item_name=item["item_name"], delivery_deadline=request["delivery_deadline"]))
            item_data = ast.literal_eval(result)
            item_data["item_delivery_deadline"] = get_supplier_delivery_date(request["delivery_deadline"], item["quantity"])
            print(f"item_data: {item_data}")
        print("#"*200)

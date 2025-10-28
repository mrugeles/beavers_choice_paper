from project_starter import InventoryAgent, model, INVENTORY_AGENT_PROMPT, INVENTORY_ITEMS

if __name__ == "__main__":
    agent = InventoryAgent(model=model)
    item_name = "Letter-sized paper"
    delivery_deadline = "2025-04-03"
    request_date = "2025-03-25"
    quantity = 700
    prompt = INVENTORY_AGENT_PROMPT.format(
        item_name=item_name,
        quantity=quantity,
        delivery_deadline=delivery_deadline,
        request_date=request_date,
        items=INVENTORY_ITEMS
    )
    response = agent.run(prompt)
    print("type(response):", type(response))
    print(response)
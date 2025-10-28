import ast

if __name__ == "__main__":
    from project_starter import OrchestrationAgent, model

    orchestrator_agent = OrchestrationAgent(model=model)
    client_request = """
    I would like to request a large order of high-quality paper supplies for an upcoming event. 
    We need 500 reams of A4 paper, 300 reams of letter-sized paper, and 200 reams of cardstock. 
    Please ensure the delivery is made by April 15, 2025. Thank you.
    """
    request_date = '2025-03-01'

    result = orchestrator_agent.call_agent(f"Structure the following quote request : {client_request}. Only return the structured quote request.")
    print(f"type of result: {type(result)}")
    print("Orchestrator agent result:", result)
    for item in result["items"]:
        print("Processing item:", item)
        item_stock = orchestrator_agent.call_agent(
            f"""
                Check stock for item: {item['item_name']} for date: {request_date}. 
                Only return the stock information as a dictionary.
            """)
        print("Item stock response:", item_stock)
        item['current_stock'] = item_stock['current_stock']
    print("Item stock information:", result)

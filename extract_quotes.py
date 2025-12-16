from project_starter import (
    model,
    structure_request,
    STRUCTURE_REQUEST_PROMPT_TEMPLATE,
    search_quote_history,
    paper_supplies
)
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
import json
import ast
from typing import Dict, List, Union, Literal
import pandas as pd

QUOTE_INFERENCE_PROMPT_TEMPLATE = """
You are an expert quoting specialist. Your task is to analyze a new quote request, compare it against historical data to find a baseline price, and then provide an improved, rounded quote with a friendly, service-oriented explanation.

You must follow these steps:
1.  Analyze the `new_quote_request`.
2.  Scan the `historical_quotes_requests` to find the most similar past jobs. Pay close attention to matches in `job_type`, `event_type`, and `order_size`.
3.  Identify the `total_amount` from the most similar historical job(s) to use as a **baseline price**.
4.  **Improve this baseline:** Apply a "bulk order" or "special" discount to this baseline (e.g., conceptually reduce it by 5-10%).
5.  **Calculate the final amount:** Take the new discounted price and **round it *down*** to the nearest "friendly" or "round" number (e.g., $93.50 becomes $90, $108 becomes $105). This is your final `inferred_total_amount`.
6.  Generate an `inferred_quote_explanation` that presents this final price to the customer.
7.  **This explanation MUST follow the style and tone of the "Explanation Style Examples" provided below.** Do NOT use the style from the `historical_quotes` explanations.

Your new explanation MUST:
* Be friendly, professional, and service-oriented.
* Start with a "Thank you" and acknowledge the order.
* Briefly list the key items and quantities from the request.
* Justify the final price by mentioning you've applied a "bulk order," "special discount," or similar concept.
* **Crucially:** State that the final `inferred_total_amount` has been **rounded down** to a "friendly price," "nice round number," or "agreeable total" to simplify budgeting for the client.

---
### Explanation Style Examples (Follow this tone and structure)

* **Example 1:** "Thank you for your large order! We have calculated the costs for 500 reams of A4 paper at $0.05 each, 300 reams of letter-sized paper at $0.06 each, and 200 reams of cardstock at $0.15 each. To reward your bulk order, we are pleased to offer a 10% discount on the total. This brings your total to a rounded and friendly price, making it easier for your budgeting needs."
* **Example 2:** "Thank you for your order! For the high-quality A4 paper, you requested 500 sheets at $0.05 each, totaling $25. The cardstock is 300 sheets at $0.15 each, totaling $45. Lastly, the 200 sheets of colored paper at $0.10 each come to $20. Since you are ordering in bulk, I've applied a special discount bringing the total cost to a nice rounded number of $85, which simplifies your budget for the upcoming performance. The total delivery will be scheduled for April 15, 2025."
* **Example 3:** "Thank you for your order! For the upcoming assembly, I've prepared a quote for 500 sheets of A4 paper, 300 sheets of colored paper, and 200 sheets of cardstock. By ordering in bulk, I've applied a discount to ensure the costs are rounded to a more agreeable total. The A4 paper and colored paper costs remain at their standard prices, while I've factored in a bulk discount on the cardstock to make the total even more appealing. This pricing approach should help us avoid feeling penny-pinched while ensuring you get the supplies you need for a successful event."

---
### Context: New Quote Request
```json
{new_quote_request_json}

### Context: Historical Quote Data
```json
{historical_quotes_json}
```
Task: Inferred Quote Response Analyze the new_quote_request against the historical_quotes_requests and provide your inferred quote response in the following JSON format. Do not include any other text or markdown formatting outside the JSON block.

{{ "inferred_total_amount": <float>, "inferred_quote_explanation": "<string>" }}
"""

@tool
def calculate_quote(structured_request:str, historical_quotes: str) -> Dict:
    """
    Tool to calculate quote based on structured request and historical quotes.
    Args:
    structured_request (dict): The structured quote request.
    historical_quotes (list): List of previous similar quotes.
    Returns:
    str: JSON string of inferred quote with total amount and explanation.
    """
    new_quote_request_json = structured_request
    historical_quotes_json = historical_quotes

    prompt = QUOTE_INFERENCE_PROMPT_TEMPLATE.format(
        new_quote_request_json=new_quote_request_json,
        historical_quotes_json=historical_quotes_json
    )
    messages = [{"role": "user", "content": prompt}]
    response = model(messages)

    try:
        # The model may return the JSON wrapped in markdown
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        return None

@tool
def get_quote_history(structured_request: dict) -> str:
    """
    Tool to search quote history for similar requests.
    Args:
    structured_request (dict): The structured quote request.
    Returns:
    str: JSON string of previous similar quotes.
    """
    items = [item['item_name'] for item in structured_request.get('items', [])]
    # Search quote history for similar requests
    previous_quotes = search_quote_history(items)

    previous_quotes_str = json.dumps(previous_quotes)
    return previous_quotes_str

def get_unit_price(item_name: str) -> float:
    """
    Tool to get the unit price of an item.
    Args:
    item_name (str): The name of the item.
    Returns:
    float: The unit price of the item.
    """
    # In a real implementation, this would query a database or pricing service
    for item in paper_supplies:
        if item['item_name'] == item_name:
            return item['unit_price']
    return 0.0

class QuoteAgent(ToolCallingAgent):
    """Agent for generating quotes."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[structure_request, get_quote_history, calculate_quote],
            model=model,
            name="quote_agent",
            description="Agent for generating quotes based on client requests.",
        )


if __name__ == "__main__":
    from project_starter import model
    quote_agent = QuoteAgent(model=model)
    request_date = "Request Date: 2025-01-01"

    df = pd.read_csv("quote_requests.csv").head(3)
    df['structured_request'] = df['response'].apply(
        lambda x: quote_agent.run(
            STRUCTURE_REQUEST_PROMPT_TEMPLATE.format(
                request_date=request_date,
                request_text=x)))
    print(df.head())
    #df.to_excel("structured_quote_requests.xlsx", index=False)




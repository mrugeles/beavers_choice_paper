import json
from smolagents import tool
from typing import Dict, List
from sqlalchemy.sql import text

from project_starter import (
    VALID_ITEMS_LIST,
    db_engine
)
from project_starter import (
    model,
    paper_supplies
)

from quote_agent.quote_agent_prompts import (
    QUOTE_INFERENCE_PROMPT_TEMPLATE
)

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

@tool
def structure_request(request_text: str, request_date: str) -> Dict:
    """
        Parses unstructured natural language client requests into a structured sales order.

        Use this tool when a user provides a raw description of an order (e.g., an email or message)
        and you need to extract specific details like delivery deadlines, item quantities, and standardized product names.

        Args:
            request_text: The raw, unstructured text containing the client's order requirements (e.g., "I need 5 banners by Friday").
            request_date: The reference date for the request in 'YYYY-MM-DD' format. Required to resolve relative dates like "next Friday" or "tomorrow".

        Returns:
            A dictionary containing keys: 'delivery_deadline', 'request_date', and a list of 'items' with mapped names and quantities.
        """
    # 2. Create a detailed prompt for the model

    prompt = f"""
        **Context:** You are an expert data extraction agent. Your task is to analyze a client request and extract 
        the delivery deadline and a list of items into a structured JSON object. You must strictly adhere to the 
        provided schema, rules, and item list.

        ### **1. JSON Output Schema**

        The final JSON object must follow this exact structure.

        ```json
        {{
          "delivery_deadline": "string (YYYY-MM-DD)",
          "request_date": "string (YYYY-MM-DD)",
          "items": [
            {{
              "item_name": "string",
              "quantity": "integer"
            }}
          ]
        }}
        ```

        ### **2. Rules and Constraints**

          * **Item Mapping:** For each item mentioned in the client's request, find the most semantically similar name from the `VALID_ITEMS_LIST` and use it for the `item_name` field.
          * **Date Formatting:** The `delivery_deadline` in the output must be in `YYYY-MM-DD` format.
          * **Missing Information:** If any field's information is not present in the request text, use `null` as its value in the JSON output.

        ### **3. Example**

        Here is an example of how to process a request correctly.

          * **VALID_ITEMS_LIST:** `["Corporate Banner", "Step-and-Repeat Backdrop", "Podium Sign", "Tablecloth", "Retractable Banner Stand"]`

          * **REQUEST_TEXT:**

            > "Hey team, we've got the annual TechGala coming up. It's a pretty big job. Order date is Oct 28, 2024. We'll need everything delivered by Nov 1, 2024. We need 3 of those big vinyl things with our logos all over it for the red carpet, and a branded cloth for the main table. Also, add 5 of those roll-up signs for the hallways."

          * **CORRECT JSON OUTPUT:**

            ```json
            {{
              "delivery_deadline": "2024-11-01",
              "request_date": "2024-10-01",
              "items": [
                {{
                  "item_name": "Step-and-Repeat Backdrop",
                  "quantity": 3
                }},
                {{
                  "item_name": "Tablecloth",
                  "quantity": 1
                }},
                {{
                  "item_name": "Retractable Banner Stand",
                  "quantity": 5
                }}
              ]
            }}
            ```

        ### **4. Your Task**

        Now, process the following client request.

          * **VALID_ITEMS_LIST:** **{VALID_ITEMS_LIST}**
          * **REQUEST_DATE: ** **{request_date}**
          * **REQUEST_TEXT:**
            > 
            > -----
            > ## **{request_text}**

        Generate the JSON object:
        """

    # 3. Call the model and get the response
    messages = [{"role": "user", "content": prompt}]
    response = model(messages)

    # 4. Clean up the response and parse the JSON
    try:
        # The model may return the JSON wrapped in markdown
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        return None


@tool
def calculate_quote(structured_request:str, historical_quotes: str) -> Dict:
    """
        Generates a financial quote and pricing explanation based on a specific order and historical pricing data.

        Use this tool AFTER you have successfully structured the client's request. It compares the current
        request against historical data to infer a consistent price.

        Args:
            structured_request (str): A JSON string representing the order details (output from the 'structure_request' tool).
                                      Must contain 'items' and 'quantity'.
            historical_quotes (str): A JSON string containing a list of past finalized quotes.
                                     Used as context to ensure the new quote aligns with previous pricing logic.

        Returns:
            Dict: A dictionary containing the calculated 'total_amount', a breakdown of costs, and an 'explanation' field describing how the price was determined.
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
def search_quote_history(search_terms: str, limit: int = 5) -> List[Dict]:
    """
    Searches the database for historical quotes matching a comma-separated list of terms.

    This tool searches both the original client request text and the final quote explanation
    for specific keywords. It returns past jobs to help benchmark pricing for new requests.

    Args:
        search_terms (str): A single string containing keywords separated by commas (e.g., "A4 paper,cardstock,banners").
        limit (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        List[Dict]: A list of historical quotes sorted by date (newest first). Each dictionary includes:
            - 'original_request': The raw text of the past client request.
            - 'total_amount': The final price of that job.
            - 'quote_explanation': The reasoning used for that price.
            - 'job_type', 'order_size', 'event_type', 'order_date'.
    """
    conditions = []
    params = {}
    print(f"DEBUG: Searching quotes with terms: {search_terms} and limit: {limit}")
    # Build SQL WHERE clause using LIKE filters for each search term
    search_terms = search_terms.split(',')
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [row._asdict() for row in result]

"""
def get_quote_history(items: str) -> str:
    '''
        Retrieves a list of historical quotes containing items similar to the current request.

        Use this tool to fetch pricing context (benchmarks) needed to calculate a new quote.
        It filters the database for past orders that match the 'item_name' fields in your structured request.

        Args:
            structured_request (dict): A dictionary representing the current order.
                                       Must strictly follow the schema: {'items': [{'item_name': '...', ...}]}.

        Returns:
            str: A JSON string containing a list of relevant historical quotes.
                 This output is intended to be passed directly into the 'calculate_quote' tool.
    '''
    # Search quote history for similar requests
    previous_quotes = search_quote_history(items)

    previous_quotes_str = json.dumps(previous_quotes)
    return previous_quotes_str
"""
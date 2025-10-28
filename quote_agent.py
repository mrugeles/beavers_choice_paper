from project_starter import structure_request, STRUCTURE_REQUEST_PROMPT_TEMPLATE, search_quote_history
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
import json
from typing import Dict, List, Union, Literal

@tool
def calculate_quote(structured_request: dict) -> str:
    """
    Calculate a quote based on a structured request.

    Args:
        structured_request (str): The structured request in JSON format.

    Returns:
        str: The quote details in JSON format.
    """
    items = [item['item_name'] for item in structured_request.get('items', [])]
    # Search quote history for similar requests
    previous_quotes = search_quote_history(items)
    print("Previous Quotes Found:", previous_quotes)

    return ""

class QuoteAgent(ToolCallingAgent):
    """Agent for generating quotes."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[structure_request, calculate_quote],
            model=model,
            name="quote_agent",
            description="Agent for generating quotes based on client requests.",
        )


if __name__ == "__main__":
    from project_starter import model

    quote_agent = QuoteAgent(model=model)
    request_date = "Request Date: 2025-04-01"
    client_request = request_date +"""
        I need to order 
        500 sheets of A4 paper, 
        300 sheets of colored paper, and 
        200 sheets of cardstock for the assembly. 
        Please deliver the supplies by April 15, 2025.
    """

    structure_prompt = STRUCTURE_REQUEST_PROMPT_TEMPLATE.format(request_date= request_date, request_text=client_request)

    result = quote_agent.run(structure_prompt)
    print("Quote Agent Result:", result)
    result = quote_agent.run("Calculate the quote: " + json.dumps(result))
    print("Final Quote Result:", result)
from quote_agent.quote_agent_tools import (
    structure_request,
    search_quote_history,
    calculate_quote,
    get_unit_price
)
from quote_agent.quote_agent_prompts import (
    STRUCTURE_REQUEST_PROMPT_TEMPLATE,
    QUOTE_HISTORY_PROMPT,
    QUOTE_REQUEST_PROMPT
)
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
import json
import ast


class QuoteAgent(ToolCallingAgent):
    """Agent for generating quotes."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[structure_request, search_quote_history, calculate_quote],
            model=model,
            name="quote_agent",
            description="Agent for generating quotes based on client requests.",
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
        with open(f"{request_path}/request.txt", "r") as f:
            client_request = f.read()

        quote_agent = QuoteAgent(model=model)
        request_date = "Request Date: 2025-01-01"
        # Structure quote request
        structure_prompt = STRUCTURE_REQUEST_PROMPT_TEMPLATE.format(request_date=request_date, request_text=client_request)
        result = quote_agent.run(structure_prompt)
        result_str = str(result)
        structured_request_data = ast.literal_eval(result_str)
        total_amount = 0.0
        for item in structured_request_data['items']:
            item_name = item['item_name']
            item['unit_price'] = get_unit_price(item_name)
            item['total_price'] = item['unit_price'] * item['quantity']
            total_amount += item['total_price']
        structured_request_data['total_amount'] = round(total_amount, 2)

        with open(f"{request_path}/request_response.json", "w") as f:
            json.dump(structured_request_data, f, indent=2)


        # Get the historical quotes (this will be a JSON string)

        items = [item['item_name'] for item in structured_request_data.get('items', [])]
        items = ','.join(items)
        historical_quotes_data_str = quote_agent.run(QUOTE_HISTORY_PROMPT.format(items=items))
        print("Quote history (Raw JSON String):", historical_quotes_data_str)

        structured_request_data_str = json.dumps(structured_request_data)



        # The agent will now infer the 'calculate_quote' tool and return raw JSON
        final_result = quote_agent.run(QUOTE_REQUEST_PROMPT.format(
            structured_request_data=structured_request_data_str,
            historical_quotes_data=historical_quotes_data_str
        ))

        final_result = str(final_result)
        final_result = ast.literal_eval(final_result)
        print(f"type(final_result): {type(final_result)}")
        print(f"final_result: {final_result}")
        quote_total = {
            "total_amount": final_result['inferred_total_amount'],
        }
        with open(f"{request_path}/quote_response.json", "w") as f:
            json.dump(quote_total, f, indent=2)
        with open(f"{request_path}/quote_response.txt", "w") as f:
            f.write(final_result['inferred_quote_explanation'])
        print("#"*200)

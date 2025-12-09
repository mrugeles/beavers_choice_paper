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

STRUCTURE_REQUEST_PROMPT_TEMPLATE = """
    Structure the following quote request: {request_text}.
    Request date: {request_date}
    Only return the structured quote request as a raw dictionary.
"""

QUOTE_HISTORY_PROMPT = """
      Call the 'search_quote_history(structured_request: str)' tool. 

      You MUST pass the string below as the 'items' argument to the tool:

      {items}

      Your final answer MUST be only the raw JSON string output from the tool.
      Do not add any conversational text, explanation, or summarization.
      """

QUOTE_REQUEST_PROMPT = """
                You are provided with two final, pre-processed pieces of data:

                1. structured_request: {structured_request_data}
                2. historical_quotes: {historical_quotes_data}

                Your task is to calculate the final inferred quote using this data directly. 
                **Do not call 'structure_request' or 'get_quote_history' again.**
                Use the provided data as-is.

                Your final answer MUST be only the raw JSON dictionary containing 
                'inferred_total_amount' and 'inferred_quote_explanation'. 
                Do not add any conversational text.
                """
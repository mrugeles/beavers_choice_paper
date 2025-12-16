import json

import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast

from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, tool


dotenv.load_dotenv()

openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)
# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

VALID_ITEMS_LIST = ','.join([item_supply["item_name"] for item_supply in paper_supplies])


# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, coverage=1, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            t.item_name,
            COALESCE(SUM(CASE
                WHEN t.transaction_type = 'stock_orders' THEN units
                WHEN t.transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock,
            i.unit_price,
            i.min_stock_level 
        FROM transactions t
        INNER JOIN inventory i on i.item_name = t.item_name 
        WHERE t.item_name = :item_name
        AND t.transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
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
    print(f"DEBUG: Executing search_quote_history with query: {query} and params: {params}")
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row) for row in result.mappings()]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
CHECK_INVENTORY_PROMPT = """
    You are a dedicated inventory execution agent.

    **Your Goal:** Run the 'check_item_stock' tool exactly once for the item: "{item_name}" on date: "{delivery_deadline}".

    **Strict Rules:**
    1. Call ONLY the 'check_item_stock' tool. Do not call 'structure_request', 'get_quote_history', or any other tool.
    2. Do not loop. Once the tool returns a result, your task is complete.
    3. Your Final Answer must be the raw JSON dictionary returned by the tool.
    4. Do NOT add any additional text, explanations, or formatting to the output.
     

    **Input Data:**
    - Item: {item_name}
    - Date: {delivery_deadline}

    Begin.
"""


PROCESS_TRANSACTION_PROMPT = """
"Please process a client transaction by calling the order_item_stock tool. Use the parameters provided in the JSON object below:"

JSON
{transaction_item}
"""
@tool
def check_item_stock(
        item_name: str,
        as_of_date: Union[str, datetime]) -> Dict:
    """
        Queries the inventory system to check stock availability for a specific item on a given date.

        Use this tool BEFORE generating a final quote or order to ensure the requested quantity
        can be fulfilled. It validates the item name against the master catalog and returns
        the quantity on hand.

        Args:
            item_name (str): The specific name of the product (e.g., "A4 Paper").
            as_of_date (str): The reference date for the inventory check in 'YYYY-MM-DD' format.

        Returns:
            Dict: A dictionary with the following keys:
                - 'item_name': The name of the item checked.
                - 'current_stock': The available quantity (int). Returns 0 if the item is invalid.
                - 'is_valid_item': Boolean indicating if the item name exists in the catalog.
    """
    if item_name in VALID_ITEMS_LIST:
        result = get_stock_level(item_name, as_of_date)
        if result["item_name"].iloc[0] is None:
            return {"item_name": item_name, "current_stock": -1, "unit_price": -1, "min_stock_level": -1}
        stock = int(result["current_stock"].iloc[0])
        unit_price = float(result["unit_price"].iloc[0])
        min_stock_level = float(result["min_stock_level"].iloc[0])
        item = {"item_name": item_name, "current_stock": stock, "unit_price": unit_price, "min_stock_level": min_stock_level}
        return item
    else:
        item = {"item_name": item_name, "current_stock": -1, "unit_price": -1, "min_stock_level": -1}
        return item

@tool
def order_item_stock(item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime]) -> int:
    """
    Place an order for a specific item and quantity, and return the transaction code.
    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.
    Returns:
        int: The transaction ID of the order.
    """
    return create_transaction(
        item_name=item_name,
        transaction_type=transaction_type,
        quantity=quantity,
        price=price,
        date=date,
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


# Tools for quoting agent

QUOTE_INFERENCE_PROMPT_TEMPLATE = """
You are an expert quoting specialist. Your task is to analyze a new quote request and provide a rounded quote with a friendly, service-oriented explanation.

You must follow these steps:
1.  **Analyze the `new_quote_request`.**
2.  **Determine the Baseline Price:**
    * Check `historical_quotes_requests` for similar past jobs (matching `job_type`, `event_type`, `order_size`).
    * If history exists, use the past `total_amount` as the baseline.
    * **If `historical_quotes_requests` is empty or irrelevant, estimate a standard competitive market price** for the items in the `new_quote_request` as your baseline.
3.  **Apply Discount:** Apply a "bulk order" or "special" discount to this baseline (conceptually reduce it by 5-10%).
4.  **Calculate Final Amount:** Take the discounted price and **round it *down*** to the nearest "friendly" or "round" number (e.g., $93.50 becomes $90). This is your `inferred_total_amount`.
5.  **Generate Explanation:** Create the `inferred_quote_explanation` strictly following the "Explanation Style Examples" below.

**CRITICAL RULES FOR EXPLANATION:**
* **Structure:** Must match the provided examples exactly.
* **Content:** Start with "Thank you," list the items/quantities, mention the discount/rounding logic, and state the final price.
* **NO METADATA:** Do NOT mention whether historical data was found or not. Do NOT say "Since there is no history..." or "Based on similar past jobs..."
* **Focus:** Focus only on the items, the discount, and the friendly final price.

---
### Explanation Style Examples (Strictly follow this tone and structure)

* **Example 1:** "Thank you for your large order! We have calculated the costs for 500 reams of A4 paper at $0.05 each, 300 reams of letter-sized paper at $0.06 each, and 200 reams of cardstock at $0.15 each. To reward your bulk order, we are pleased to offer a 10% discount on the total. This brings your total to a rounded and friendly price, making it easier for your budgeting needs."
* **Example 2:** "Thank you for your order! For the high-quality A4 paper, you requested 500 sheets at $0.05 each, totaling $25. The cardstock is 300 sheets at $0.15 each, totaling $45. Lastly, the 200 sheets of colored paper at $0.10 each come to $20. Since you are ordering in bulk, I've applied a special discount bringing the total cost to a nice rounded number of $85, which simplifies your budget for the upcoming performance. The total delivery will be scheduled for April 15, 2025."
* **Example 3:** "Thank you for your order! For the upcoming assembly, I've prepared a quote for 500 sheets of A4 paper, 300 sheets of colored paper, and 200 sheets of cardstock. By ordering in bulk, I've applied a discount to ensure the costs are rounded to a more agreeable total. The A4 paper and colored paper costs remain at their standard prices, while I've factored in a bulk discount on the cardstock to make the total even more appealing. This pricing approach should help us avoid feeling penny-pinched while ensuring you get the supplies you need for a successful event."

---
### Context: New Quote Request
```json
{new_quote_request_json}
```

###Context: Historical Quote Data
```json
{historical_quotes_json}
```

###Task: Inferred Quote Response Analyze the new_quote_request and (optionally) the historical_quotes_requests to provide your inferred quote response in the following JSON format. Do not include any other text or markdown formatting outside the JSON block.

{{
"inferred_total_amount": <float>,
"inferred_quote_explanation": "<string>"
}}
"""

STRUCTURE_REQUEST_PROMPT_TEMPLATE = """
    Structure the following quote request.
    * **REQUEST_TEXT:**
            > 
            > -----
            > ## **{request_text}**
    * **REQUEST_DATE: ** **{request_date}**
    
    - Call only the 'structure_request(request_text: str, request_date: str)' tool.
    - Only return the structured quote request as a raw dictionary.
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
def get_quote_history(search_terms: List[str]) -> List[Dict]:
    """
        Retrieve a list of historical quotes that match any of the provided search terms.

        The function searches both the original customer request (from `quote_requests`) and
        the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
        most recent order date and limited by the `limit` parameter.

        Args:
            search_terms (List[str]): List of terms to match against customer requests and explanations.
            limit (int, optional): Maximum number of quote records to return. Default is 5.

        Returns:
            List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
                - original_request
                - total_amount
                - quote_explanation
                - job_type
                - order_size
                - event_type
                - order_date
    """
    return search_quote_history(search_terms, limit=5)

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
def calculate_quote_no_discount(structured_request:str, historical_quotes: str) -> Dict:
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
class QuoteAgent(ToolCallingAgent):
    """Agent for generating quotes."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[structure_request, get_quote_history, calculate_quote],
            model=model,
            name="quote_agent",
            description="Agent for generating quotes based on client requests.",
        )

# Tools for ordering agent


# Set up your agents and create an orchestration agent that will manage them.
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
class OrchestratorAgent(ToolCallingAgent):
    """Agent for generating quotes."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="orchestrator_agent",
            description="Orchestrates the full lifecycle of a client quote request, from data extraction and "
                        "inventory validation to final price calculation."
        )
        self.quote_agent = QuoteAgent(model=model)
        self.inventory_agent = InventoryAgent(model=model)

    def process_quote_request(self, request: str, request_date: str) -> str:
        response = None
        format_string = "%Y-%m-%d"
        # Step 1: Structure the request
        request = self.structure_quote_request(request, request_date)
        # Check stock
        for item in request["items"]:
            delivery_deadline = datetime.strptime(request["delivery_deadline"], format_string)
            item = self.get_item_inventory(format_string, item, request)

            if (item["current_stock"] < item["quantity"]) and (item["supplier_delivery_date"] > delivery_deadline):
                print("Could not process quote due to insufficient stock.")
                return "Could not process quote due to insufficient stock."

            if item["supplier_delivery_date"] is not None:
                item["supplier_delivery_date"] = item["supplier_delivery_date"].strftime(format_string)
        # Create quote
        ## Get the historical quotes (this will be a JSON string)
        items = [item['item_name'] for item in request.get('items', [])]
        items = ','.join(items)
        historical_quotes_data_str = self.quote_agent.run(QUOTE_HISTORY_PROMPT.format(items=items))

        structured_request_data_str = json.dumps(request)
        # The agent will now infer the 'calculate_quote' tool and return raw JSON
        final_result = self.quote_agent.run(QUOTE_REQUEST_PROMPT.format(
            structured_request_data=structured_request_data_str,
            historical_quotes_data=historical_quotes_data_str
        ))

        final_result = str(final_result)
        final_result = ast.literal_eval(final_result)
        print(f"final_result: {final_result}")
        print(f"request: {request}")
        response = final_result["inferred_quote_explanation"]
        for item in request["items"]:
            transaction_item = {
                "item_name": item["item_name"],
                "transaction_type": "sales",
                "quantity": item["quantity"],
                "price": get_unit_price(item["item_name"]) * item["quantity"],
                "date": request["request_date"]
            }
            if item["current_stock"] == -1:
                transaction_item["quantity"] = 0
            # Sales transaction
            transaction_id = self.inventory_agent.run(PROCESS_TRANSACTION_PROMPT.format(transaction_item=transaction_item))
            print(f"transaction_id: {transaction_id}")
            item = self.get_item_inventory(format_string, item, request)
            if item["current_stock"] < item["min_stock_level"]:
                order_quantity = item["min_stock_level"] * 2 - item["current_stock"]
                order_transaction_item = {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "quantity": order_quantity,
                    "price": get_unit_price(item["item_name"]) * order_quantity,
                    "date": request["request_date"]
                }
                # Order stock transaction
                order_transaction_id = self.inventory_agent.run(PROCESS_TRANSACTION_PROMPT.format(transaction_item=order_transaction_item))
                print(f"order_transaction_id: {order_transaction_id}")


        # Sales transaction
        # Update stock
        # Check stock
        ## stock_orders for items with quantity < min stock

        return response

    def get_item_inventory(self, format_string, item, request):


        item_data = self.inventory_agent.run(CHECK_INVENTORY_PROMPT.format(item_name=item["item_name"],
                                                                                delivery_deadline=request[
                                                                                    "delivery_deadline"]))
        supplier_delivery_date_str = get_supplier_delivery_date(request["request_date"], item["quantity"])
        supplier_delivery_date = datetime.strptime(supplier_delivery_date_str, format_string)
        item["current_stock"] = item_data["current_stock"]
        item["min_stock_level"] = item_data["min_stock_level"]
        item["supplier_delivery_date"] = supplier_delivery_date
        print(f"item: {item}")

        return item

    def structure_quote_request(self, request, request_date):
        structure_prompt = STRUCTURE_REQUEST_PROMPT_TEMPLATE.format(request_date=request_date,
                                                                    request_text=request)
        result = self.quote_agent.run(structure_prompt)
        result_str = str(result)
        structured_request_data = ast.literal_eval(result_str)
        total_amount = 0.0
        for item in structured_request_data['items']:
            item_name = item['item_name']
            item['unit_price'] = get_unit_price(item_name)
            item['total_price'] = item['unit_price'] * item['quantity']
            total_amount += item['total_price']
        structured_request_data['total_amount'] = round(total_amount, 2)
        return structured_request_data


# Run your test scenarios by writing them here. Make sure to keep track of them.

def call_multi_agent_system(row):
    response = None
    format_string = "%Y-%m-%d"
    orchestrator_agent = OrchestratorAgent(model=model)

    response = orchestrator_agent.process_quote_request(row["request"], row["request_date"].strftime(format_string))
    return response


def get_build_stock_status(format_string, inventory_agent, item, request, transaction_items):
    delivery_deadline = datetime.strptime(request["delivery_deadline"], format_string)
    result = inventory_agent.run(
        CHECK_INVENTORY_PROMPT.format(item_name=item["item_name"], delivery_deadline=request["delivery_deadline"]))
    item_data = ast.literal_eval(result)
    item_delivery_deadline_str = get_supplier_delivery_date(request["delivery_deadline"], item["quantity"])
    item_delivery_deadline = datetime.strptime(item_delivery_deadline_str, format_string)
    item_data["item_delivery_deadline"] = item_delivery_deadline
    transaction_item = {
        "item_name": item["item_name"],
        "transaction_type": "sales",
        "quantity": item["quantity"],
        "price": item_data["unit_price"] * item["quantity"],
        "date": item_delivery_deadline_str
    }
    if item_delivery_deadline > delivery_deadline or item_data["current_stock"] < item["quantity"]:
        transaction_item["quantity"] = -1
    return transaction_item


def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

    # Sort by date
    quote_requests_sample["request_date"] = pd.to_datetime(
        quote_requests_sample["request_date"]
    )
    quote_requests_sample = quote_requests_sample.sort_values("request_date")

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = call_multi_agent_system(row)
        #response = None
        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
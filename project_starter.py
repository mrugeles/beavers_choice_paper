import time

import pandas as pd
import numpy as np
import os
import json
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Literal

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
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
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
    print(f"DEBUG: Searching quotes with terms: {search_terms} and limit: {limit}")
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
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [row._asdict() for row in result]

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
@tool
def check_item_stock(
        item_name: str,
        as_of_date: Union[str, datetime]) -> Dict:
    """
    Check the current stock level of a specific item as of a given date.
    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date (in ISO format) to check stock as of.
        task (str): The task context, either 'structure_request_only' or 'full_quote_calculation'.
    Returns:
        int: The current stock level of the item.
    """
    if item_name in VALID_ITEMS_LIST:
        stock = get_stock_level(item_name, as_of_date)
        print(f"DEBUG: Stock DataFrame for '{item_name}' as of {as_of_date}:\n{stock}")
        stock = int(stock["current_stock"].iloc[0])
        print(f"DEBUG: Current stock of '{item_name}' as of {as_of_date} is {stock}")
        return {"item_name": item_name, "current_stock": stock, "is_valid_item": True}
    else:
        return {"item_name": item_name, "current_stock": 0, "is_valid_item": False}


@tool
def check_item_min_stock(item_name: str) -> int:
    """
    Check the current stock level of a specific item as of a given date.
    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date (in ISO format) to check stock as of.
    Returns:
        int: The current stock level of the item.
    """
    query = """
            SELECT
            min_stock_level  
            FROM inventory 
            WHERE
            item_name = :item_name
        """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"item_name": item_name})
    min_stock_level = int(result["min_stock_level"].iloc[0])
    # Convert the result into a dictionary {item_name: stock}
    return min_stock_level

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

@tool
def get_item_delivery_date(order_date: str, quantity: int) -> str:
    """
    Get the estimated delivery date from the supplier for a specific item and quantity based on the order date.
    Args:
        quantity (int): The number of units ordered.
        order_date (str): The date the order was placed (in ISO format).
    Returns:
        str: The estimated delivery date (in ISO format).
    """
    return get_supplier_delivery_date(order_date, quantity)


@tool
def get_item_unit_price(item_name: str) -> float:
    """
    Get the unit price of a specific item from the inventory.
    Args:
        item_name (str): The name of the item.
    Returns:
        float: The unit price of the item.
    """
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    item_row = inventory_df[inventory_df["item_name"] == item_name]
    if not item_row.empty:
        return float(item_row.iloc[0]["unit_price"])
    else:
        raise ValueError(f"Item '{item_name}' not found in inventory.")

# Tools for quoting agent
VALID_ITEMS_LIST = ','.join([item_supply["item_name"] for item_supply in paper_supplies])

@tool
def structure_request(request_text: str) -> Dict:
    """
        Uses a language model to convert a raw quote request into a structured JSON object.

        Args:
            request_text: The raw text from a client's request.

        Returns:
            A dictionary representing the structured request.
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


#

# Tools for ordering agent


# Set up your agents and create an orchestration agent that will manage them.
"""
class QuoteAgent(ToolCallingAgent):
   ---Agent for generating quotes.---

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[structure_request, calculate_quote],
            model=model,
            name="quote_agent",
            description="Agent for generating quotes based on client requests.",
        )
"""
class InventoryAgent(ToolCallingAgent):
    """Agent for managing inventory."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[check_item_stock, order_item_stock, get_item_delivery_date, get_item_unit_price],
            model=model,
            name="inventory_agent",
            description="Agent for managing inventory. Check stock levels and order stock items.",
        )


INVENTORY_AGENT_PROMPT = """
        # Request
            - Client requires {quantity}  of {item_name} by {delivery_deadline}.
            - Request date is {request_date}.
        # Tasks to perform
        1) Get item unit price from inventory.
        2) Get current stock level from inventory.
        3) If stock level less than quantity:
            - Order the difference for {item_name} (quantity - stock level).
            - Set the delivery date for {item_name} as the provided by the supplier.
            - Set is_supplier_delivery to True.
        4) If stock levels more than the quantity or equal:
            - Set delivery date as the one asked by the client.
            - Set is_supplier_delivery to False.
        6) Return item, quantity asked by client, unit price and delivery date.   

        Format your response as a JSON object with the following keys:
        {{
            "item_name": str,
            "quantity": int,
            "unit_price": float,
            "total_price": float,
            "delivery_date": str (ISO format),
            "is_supplier_delivery": bool
        }}     
        """

INVENTORY_ITEMS = ','.join([item_supply["item_name"] for item_supply in paper_supplies])



# project_starter.py

@tool
def call_inventory_agent(
    request_text: str,
    task: Literal["check_item_stock_only"] = "check_item_stock_only"
) -> Dict:
    """
    Calls the Inventory Agent to process a request.
    Use 'check_item_stock_only' to check stock levels for items in the request.
    Args:
        request_text (str): The raw text from a client's request.
        task (str): The specific task for the Quote Agent to perform.
    """
    inventory_agent = InventoryAgent(model)
    prompt = f"""
        Your task is to perform: '{task}'.
        Here is the request:
        ---
        {request_text}
        ---
        Execute the necessary tool. Your final answer MUST be only the raw dictionary output
        from that tool call. Do not include any other text, markdown, or explanation.
        """
    return inventory_agent.run(prompt)
@tool
def call_quote_agent(
    request_text: str,
    task: Literal["structure_request_only", "full_quote_calculation"] = "full_quote_calculation"
) -> Dict:
    """
    Calls the Quote Agent to process a client request.
    Use 'structure_request_only' to parse the request into JSON.
    Use 'full_quote_calculation' to structure the request AND calculate the final price.

    Args:
        request_text (str): The raw text from a client's request.
        task (str): The specific task for the Quote Agent to perform.
    quote_agent = QuoteAgent(model)

    # Create a more specific prompt for the QuoteAgent that includes the task
    prompt = ---
        Your task is to perform: '{task}'.
        Here is the client's request:
        ---
        {request_text}
        ---
        Execute the task and respond with the final result.
    ---
    return quote_agent.run(prompt)
    """

class OrchestrationAgent(ToolCallingAgent):
    """Agent for orchestrating the multi-agent system."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[ call_inventory_agent],
            model=model,
            name="orchestration_agent",
            description="Agent for orchestrating the multi-agent system.",
        )
        self.model = model

    def call_agent(self, request_text: str) -> Dict:
        result = self.run(request_text)
        result_str = str(result)
        result_dict = ast.literal_eval(result_str)
        return result_dict

    """
    def call_inventory_agent(self, item_name, quantity, delivery_deadline, request_date) -> str:
        prompt = INVENTORY_AGENT_PROMPT.format(item_name=item_name, quantity=quantity, delivery_deadline=delivery_deadline, request_date=request_date, items=INVENTORY_ITEMS)
        return self.inventory_agent.run(prompt)
    """
# Initialize database engine
# Run your test scenarios by writing them here. Make sure to keep track of them.

# project_starter.py

# --- Prompt Templates ---
STRUCTURE_REQUEST_PROMPT_TEMPLATE = """
    Structure the following quote request: {request_text}.
    Request date: {request_date}
    Only return the structured quote request as a raw dictionary.
"""

CHECK_STOCK_PROMPT_TEMPLATE = """
    Use the inventory agent to check the stock for item '{item_name}' on date {request_date}.
    Your final answer MUST be the raw, unmodified dictionary that the inventory agent returns.
    Do not change, reformat, or summarize it in any way.
"""
import time

import pandas as pd
import numpy as np
import os
import json
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Literal

from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

dotenv.load_dotenv()

#openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-5-mini",
    #api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
    #temperature=0.2
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
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
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
    print(f"DEBUG: Searching quotes with terms: {search_terms} and limit: {limit}")
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
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [row._asdict() for row in result]

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
@tool
def check_item_stock(
        item_name: str,
        as_of_date: Union[str, datetime]) -> Dict:
    """
    Check the current stock level of a specific item as of a given date.
    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date (in ISO format) to check stock as of.
        task (str): The task context, either 'structure_request_only' or 'full_quote_calculation'.
    Returns:
        int: The current stock level of the item.
    """
    if item_name in VALID_ITEMS_LIST:
        stock = get_stock_level(item_name, as_of_date)
        print(f"DEBUG: Stock DataFrame for '{item_name}' as of {as_of_date}:\n{stock}")
        stock = int(stock["current_stock"].iloc[0])
        print(f"DEBUG: Current stock of '{item_name}' as of {as_of_date} is {stock}")
        return {"item_name": item_name, "current_stock": stock, "is_valid_item": True}
    else:
        return {"item_name": item_name, "current_stock": 0, "is_valid_item": False}


@tool
def check_item_min_stock(item_name: str) -> int:
    """
    Check the current stock level of a specific item as of a given date.
    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date (in ISO format) to check stock as of.
    Returns:
        int: The current stock level of the item.
    """
    query = """
            SELECT
            min_stock_level  
            FROM inventory 
            WHERE
            item_name = :item_name
        """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"item_name": item_name})
    min_stock_level = int(result["min_stock_level"].iloc[0])
    # Convert the result into a dictionary {item_name: stock}
    return min_stock_level

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

@tool
def get_item_delivery_date(order_date: str, quantity: int) -> str:
    """
    Get the estimated delivery date from the supplier for a specific item and quantity based on the order date.
    Args:
        quantity (int): The number of units ordered.
        order_date (str): The date the order was placed (in ISO format).
    Returns:
        str: The estimated delivery date (in ISO format).
    """
    return get_supplier_delivery_date(order_date, quantity)


@tool
def get_item_unit_price(item_name: str) -> float:
    """
    Get the unit price of a specific item from the inventory.
    Args:
        item_name (str): The name of the item.
    Returns:
        float: The unit price of the item.
    """
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    item_row = inventory_df[inventory_df["item_name"] == item_name]
    if not item_row.empty:
        return float(item_row.iloc[0]["unit_price"])
    else:
        raise ValueError(f"Item '{item_name}' not found in inventory.")

# Tools for quoting agent
VALID_ITEMS_LIST = ','.join([item_supply["item_name"] for item_supply in paper_supplies])

@tool
def structure_request(request_text: str) -> Dict:
    """
        Uses a language model to convert a raw quote request into a structured JSON object.

        Args:
            request_text: The raw text from a client's request.

        Returns:
            A dictionary representing the structured request.
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


#

# Tools for ordering agent


# Set up your agents and create an orchestration agent that will manage them.
"""
class QuoteAgent(ToolCallingAgent):
   ---Agent for generating quotes.---

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[structure_request, calculate_quote],
            model=model,
            name="quote_agent",
            description="Agent for generating quotes based on client requests.",
        )
"""
class InventoryAgent(ToolCallingAgent):
    """Agent for managing inventory."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[check_item_stock, order_item_stock, get_item_delivery_date, get_item_unit_price],
            model=model,
            name="inventory_agent",
            description="Agent for managing inventory. Check stock levels and order stock items.",
        )


INVENTORY_AGENT_PROMPT = """
        # Request
            - Client requires {quantity}  of {item_name} by {delivery_deadline}.
            - Request date is {request_date}.
        # Tasks to perform
        1) Get item unit price from inventory.
        2) Get current stock level from inventory.
        3) If stock level less than quantity:
            - Order the difference for {item_name} (quantity - stock level).
            - Set the delivery date for {item_name} as the provided by the supplier.
            - Set is_supplier_delivery to True.
        4) If stock levels more than the quantity or equal:
            - Set delivery date as the one asked by the client.
            - Set is_supplier_delivery to False.
        6) Return item, quantity asked by client, unit price and delivery date.   

        Format your response as a JSON object with the following keys:
        {{
            "item_name": str,
            "quantity": int,
            "unit_price": float,
            "total_price": float,
            "delivery_date": str (ISO format),
            "is_supplier_delivery": bool
        }}     
        """

INVENTORY_ITEMS = ','.join([item_supply["item_name"] for item_supply in paper_supplies])



# project_starter.py

@tool
def call_inventory_agent(
    request_text: str,
    task: Literal["check_item_stock_only"] = "check_item_stock_only"
) -> Dict:
    """
    Calls the Inventory Agent to process a request.
    Use 'check_item_stock_only' to check stock levels for items in the request.
    Args:
        request_text (str): The raw text from a client's request.
        task (str): The specific task for the Quote Agent to perform.
    """
    inventory_agent = InventoryAgent(model)
    prompt = f"""
        Your task is to perform: '{task}'.
        Here is the request:
        ---
        {request_text}
        ---
        Execute the necessary tool. Your final answer MUST be only the raw dictionary output
        from that tool call. Do not include any other text, markdown, or explanation.
        """
    return inventory_agent.run(prompt)
@tool
def call_quote_agent(
    request_text: str,
    task: Literal["structure_request_only", "full_quote_calculation"] = "full_quote_calculation"
) -> Dict:
    """
    Calls the Quote Agent to process a client request.
    Use 'structure_request_only' to parse the request into JSON.
    Use 'full_quote_calculation' to structure the request AND calculate the final price.

    Args:
        request_text (str): The raw text from a client's request.
        task (str): The specific task for the Quote Agent to perform.
    quote_agent = QuoteAgent(model)

    # Create a more specific prompt for the QuoteAgent that includes the task
    prompt = ---
        Your task is to perform: '{task}'.
        Here is the client's request:
        ---
        {request_text}
        ---
        Execute the task and respond with the final result.
    ---
    return quote_agent.run(prompt)
    """

class OrchestrationAgent(ToolCallingAgent):
    """Agent for orchestrating the multi-agent system."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[ call_inventory_agent],
            model=model,
            name="orchestration_agent",
            description="Agent for orchestrating the multi-agent system.",
        )
        self.model = model

    def call_agent(self, request_text: str) -> Dict:
        result = self.run(request_text)
        result_str = str(result)
        result_dict = ast.literal_eval(result_str)
        return result_dict

    """
    def call_inventory_agent(self, item_name, quantity, delivery_deadline, request_date) -> str:
        prompt = INVENTORY_AGENT_PROMPT.format(item_name=item_name, quantity=quantity, delivery_deadline=delivery_deadline, request_date=request_date, items=INVENTORY_ITEMS)
        return self.inventory_agent.run(prompt)
    """
# Initialize database engine
# Run your test scenarios by writing them here. Make sure to keep track of them.

# project_starter.py

# --- Prompt Templates ---
STRUCTURE_REQUEST_PROMPT_TEMPLATE = """
    Structure the following quote request: {request_text}.
    Request date: {request_date}
    Only return the structured quote request as a raw dictionary.
"""

CHECK_STOCK_PROMPT_TEMPLATE = """
    Use the inventory agent to check the stock for item '{item_name}' on date {request_date}.
    Your final answer MUST be the raw, unmodified dictionary that the inventory agent returns.
    Do not change, reformat, or summarize it in any way.
"""
def call_multi_agent_system(request_text: str) -> str:

    orchestrator_agent = OrchestrationAgent(model=model)
    structure_prompt = STRUCTURE_REQUEST_PROMPT_TEMPLATE.format(request_text=request_text)
    result = orchestrator_agent.call_agent(structure_prompt)
    print("Structured request:", result)
    for item in result["items"]:
        print("Processing item:", item)
        stock_prompt = CHECK_STOCK_PROMPT_TEMPLATE.format(
            item_name=item['item_name'],
            request_date=result['request_date']
        )
        item_stock = orchestrator_agent.call_agent(stock_prompt)
        print("Item stock response:", item_stock)
        #item['current_stock'] = item_stock['current_stock']
        #item["is_valid_item"] = item_stock["is_valid_item"]
        if not item_stock["is_valid_item"]:
            print(f"Item '{item['item_name']}' is not a valid item.")
            return "Error parsing response."
        if item_stock["current_stock"] < item['quantity']:
            print(f"Item '{item['item_name']}' is out of stock.")
            return "Error parsing response."
        item["unit_price"] = get_item_unit_price(item['item_name'])
    print("Item stock information:", result)
    return ""

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)

    request_with_date = """
        I would like to request the following paper supplies for the ceremony:                                                                                                                                                  │
                                                                                                                                                                                                                                         │
         - 100 sheets of A4 glossy paper                                                                                                                                                                                                 │
         - 100 sheets of heavy cardstock (white)                                                                                                                                                                                         │
         - 100 sheets of colored paper (assorted colors)
                                                                                                                                                                                                                                         │
         I need these supplies delivered by April 15, 2025. Thank you. (Date of request: 2025-04-01). 

    response = call_multi_agent_system(request_with_date)
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

    # quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

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

        response = call_multi_agent_system(request_with_date)
        #response = '{"item_name": "A4 paper", "units_requested": 500, "unit_price": 0.05, "total_price": 25.0, "delivery_date": "2025-04-29", "is_supplier_delivery": true}'
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
    """
    return None


if __name__ == "__main__":
    results = run_test_scenarios()
def call_multi_agent_system(request_text: str) -> str:

    orchestrator_agent = OrchestrationAgent(model=model)
    structure_prompt = STRUCTURE_REQUEST_PROMPT_TEMPLATE.format(request_text=request_text)
    result = orchestrator_agent.call_agent(structure_prompt)
    print("Structured request:", result)
    for item in result["items"]:
        print("Processing item:", item)
        stock_prompt = CHECK_STOCK_PROMPT_TEMPLATE.format(
            item_name=item['item_name'],
            request_date=result['request_date']
        )
        item_stock = orchestrator_agent.call_agent(stock_prompt)
        print("Item stock response:", item_stock)
        #item['current_stock'] = item_stock['current_stock']
        #item["is_valid_item"] = item_stock["is_valid_item"]
        if not item_stock["is_valid_item"]:
            print(f"Item '{item['item_name']}' is not a valid item.")
            return "Error parsing response."
        if item_stock["current_stock"] < item['quantity']:
            print(f"Item '{item['item_name']}' is out of stock.")
            return "Error parsing response."
        item["unit_price"] = get_item_unit_price(item['item_name'])
    print("Item stock information:", result)
    return ""

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)

    request_with_date = """
        I would like to request the following paper supplies for the ceremony:                                                                                                                                                  │
                                                                                                                                                                                                                                         │
         - 100 sheets of A4 glossy paper                                                                                                                                                                                                 │
         - 100 sheets of heavy cardstock (white)                                                                                                                                                                                         │
         - 100 sheets of colored paper (assorted colors)
                                                                                                                                                                                                                                         │
         I need these supplies delivered by April 15, 2025. Thank you. (Date of request: 2025-04-01). 

    response = call_multi_agent_system(request_with_date)
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

    # quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

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

        response = call_multi_agent_system(request_with_date)
        #response = '{"item_name": "A4 paper", "units_requested": 500, "unit_price": 0.05, "total_price": 25.0, "delivery_date": "2025-04-29", "is_supplier_delivery": true}'
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
    """
    return None


if __name__ == "__main__":
    results = run_test_scenarios()
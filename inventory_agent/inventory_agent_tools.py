from datetime import datetime, timedelta
from typing import Dict, List, Union
from smolagents import tool
import pandas as pd
from project_starter import (
    VALID_ITEMS_LIST,
    get_stock_level,
    db_engine,
    create_transaction,
    get_supplier_delivery_date
)

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
        #print(f"DEBUG: Stock DataFrame for '{item_name}' as of {as_of_date}:\n{result}")
        stock = int(result["current_stock"].iloc[0])
        unit_price = float(result["unit_price"].iloc[0])
        min_stock_level = float(result["min_stock_level"].iloc[0])
        #print(f"DEBUG: Current stock of '{item_name}' as of {as_of_date} is {stock}")
        return {"item_name": item_name, "current_stock": stock, "unit_price": unit_price, "min_stock_level": min_stock_level}
    else:
        return {"item_name": item_name, "current_stock": -1, "unit_price": -1, "min_stock_level": -1}


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



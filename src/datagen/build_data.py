import os
import json
import numpy as np
import pandas as pd

from datagen.consumers import generate_consumers
from datagen.products import generate_products
from datagen.transactions import generate_transactions


def generate_and_read_data(out_dir: str, n_consumers: int, n_products: int, seed: int = 10) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic consumers, products, and transactions; save to CSV; return DataFrames.

    Behavior:
      - If all three CSVs already exist under `out_dir`, load and return them (idempotent).
      - Otherwise, generate the data, save to CSV, and return the in-memory DataFrames.

    Notes:
      - Dates are parsed on load (consumers.start_date, transactions.transaction_date).
      - products.available_countries is stored as JSON per row to preserve list structure.

    Args:
        out_dir: Folder where CSVs live (created if missing).
        n_consumers: Number of consumers to generate (used only when creating).
        n_products: Number of products to generate (used only when creating).
        seed: NumPy RNG seed for reproducibility (used only when creating).

    Returns:
        Dict with keys "consumers", "products", "transactions" mapping to DataFrames.
    """
    consumers_path    = os.path.join(out_dir, "consumers.csv")
    products_path     = os.path.join(out_dir, "products.csv")
    transactions_path = os.path.join(out_dir, "transactions.csv")

    # If data already exists, load & return it
    if all(os.path.exists(p) for p in [consumers_path, products_path, transactions_path]):
        consumers_df = pd.read_csv(consumers_path, parse_dates=["start_date"])
        products_df = pd.read_csv(products_path, converters={"available_countries": json.loads})
        transactions_df = pd.read_csv(transactions_path, parse_dates=["transaction_date"])
        return {"consumers": consumers_df, "products": products_df, "transactions": transactions_df}

    # Otherwise, generate and save
    np.random.seed(seed)

    consumers_df = generate_consumers(n_consumers)
    products_df = generate_products(n_products)
    transactions_df = generate_transactions(consumers_df, products_df)

    # Save (CSV only); serialize list columns as JSON strings
    os.makedirs(out_dir, exist_ok=True)

    consumers_df.to_csv(consumers_path, index=False, encoding="utf-8")

    products_to_save = products_df.copy()
    products_to_save["available_countries"] = products_to_save["available_countries"].apply(json.dumps)
    products_to_save.to_csv(products_path, index=False, encoding="utf-8")

    transactions_df.to_csv(transactions_path, index=False, encoding="utf-8")

    return {"consumers": consumers_df, "products": products_df, "transactions": transactions_df}

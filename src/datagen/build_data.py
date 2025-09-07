import os, json, hashlib
from datetime import datetime
import numpy as np
import pandas as pd


from datagen.consumers import generate_consumers
from datagen.products import generate_products
from datagen.transactions import generate_transactions


def build_and_save_synth(out_dir, n_consumers, n_products, seed = 10, file_format = "csv"):
    """
    Generate synthetic consumers, products, and transactions; save to disk; return DataFrames.
    Designed to be called from a notebook (no CLI required).
    """
    np.random.seed(seed)

    # 1) Generate base tables
    consumers_df  = generate_consumers(n_consumers)
    products_df   = generate_products(n_products)

    # 2) Generate transactions table
    transactions_df = generate_transactions(consumers_df, products_df)

    # 3) Save
    extsave = "to_csv"
    save_kwargs = {"index": False}
    save_kwargs["encoding"] = "utf-8"

    getattr(consumers_df, extsave)(os.path.join(out_dir, f"consumers.{file_format}"), **save_kwargs)
    getattr(products_df,  extsave)(os.path.join(out_dir, f"products.{file_format}"),  **save_kwargs)
    getattr(transactions_df, extsave)(os.path.join(out_dir, f"transactions.{file_format}"), **save_kwargs)

    return {"consumers": consumers_df, "products": products_df, "transactions": transactions_df}

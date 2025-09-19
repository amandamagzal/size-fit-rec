"""Transaction generation utilities.

Functions:
- generate_purchases(max_purchases): draw a realistic purchase count per consumer
- generate_sequential_purchases(start_date, num_purchases): sequential timestamps
- sample_purchased_size(consumer, product_type): sampled purchased size per product type
- calculate_fit(consumer, product, product_type, purchased_size): fit outcome label
- generate_transactions(consumers_df, products_df): full transactions table
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from datagen.constants import SIZES, SHOE_SIZES


def generate_purchases(max_purchases: int = 3000) -> int:
    """Sample a realistic number of purchases for a consumer.

    Uses a truncated normal distribution with a small probability mass at 1.

    Args:
        max_purchases: Upper bound for purchases (inclusive).

    Returns:
        Integer number of purchases (≥1).
    """
    mean = 5
    std_dev = 2.0
    lower, upper = 1, max_purchases
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    base_sample = truncnorm(a, b, loc=mean, scale=std_dev).rvs()
    purchases = int(round(base_sample))

    # Clamp to valid range
    purchases = max(lower, min(purchases, upper))

    # Occasionally collapse to exactly one purchase
    if purchases != 1 and np.random.rand() < 0.1:
        purchases = 1

    return purchases


def generate_sequential_purchases(start_date: pd.Timestamp, num_purchases: int) -> list[pd.Timestamp]:
    """Generate sequential purchase dates starting from `start_date`.

    Each next purchase occurs 1–89 days after the previous one.

    Args:
        start_date: First purchase date.
        num_purchases: Number of purchase timestamps to generate.

    Returns:
        List of purchase timestamps (length == num_purchases).
    """
    purchase_dates = []
    current = start_date

    for _ in range(num_purchases):
        # Randomized days between purchases
        days_between = np.random.randint(1, 90)
        current = current + timedelta(days=days_between)
        purchase_dates.append(current)

    return purchase_dates


def sample_purchased_size(consumer: pd.Series, product_type: str):
    """Sample a purchased size near the consumer’s true size.

    Args:
        consumer: Consumer row with 'upper_size', 'lower_size', 'shoe_size'.
        product_type: Product type string (contains 'Shoes' or 'Pants & Leggings' etc.).

    Returns:
        Purchased size (string for apparel, float for shoes).
    """
    # Choose the relevant size scale and the consumer’s true size
    if "Shoes" in product_type:
        size_list = SHOE_SIZES
        true_size = consumer["shoe_size"]
    elif "Pants & Leggings" in product_type:
        size_list = SIZES
        true_size = consumer["lower_size"]
    else:
        size_list = SIZES
        true_size = consumer["upper_size"]

    # Current index of the true size
    current_index = np.where(np.array(size_list) == true_size)[0][0]

    # Mostly buy true size, sometimes one size off
    size_variation = np.random.choice([-1, 0, 1], p = [0.15, 0.7, 0.15])
    new_index = current_index + size_variation

    # Clamp to valid range
    if new_index > (len(size_list) - 1):
        return size_list[-1]
    elif new_index < 0:
        return size_list[0]
    else:
        return size_list[new_index]


def calculate_fit(
    consumer: pd.Series,
    product: pd.Series,
    product_type: str,
    purchased_size,
) -> str:
    """Compute the fit outcome based on consumer tolerances and product fit offset.

    Logic converts apparel sizes to an ordinal scale and compares purchased size
    against the consumer’s adjusted true size (true size + product fit offset),
    with personal tolerances applied.

    Args:
        consumer: Consumer row with size/tolerance fields.
        product: Product row with 'fit_offset'.
        product_type: Product type string (to decide upper/lower/shoe sizing).
        purchased_size: Purchased size (string for apparel, float for shoes).

    Returns:
        One of: 'too small', 'too large', 'fit', 'not applicable'.
    """
    size_mapping = {"2XS": 1, "XS": 2, "S": 3, "M": 4, "L": 5, "XL": 6, "2XL": 7}

    # Determine true size
    if "Shoes" in product_type:
        true_size = consumer["shoe_size"]
    elif "Pants & Leggings" in product_type:
        true_size = size_mapping[consumer["lower_size"]]
    else:
        true_size = size_mapping[consumer["upper_size"]]

    # Convert purchased size if needed
    if isinstance(purchased_size, str):
        purchased_size = size_mapping[purchased_size]

    # Adjustments and tolerances
    tol_small = consumer["consumer_tolerance_too_small"]
    tol_large = consumer["consumer_tolerance_too_large"]
    fit_offset = product["fit_offset"]

    adjusted_size = true_size + fit_offset

    if purchased_size < adjusted_size + tol_small:
        return "too small"
    elif purchased_size > adjusted_size + tol_large:
        return "too large"
    elif (true_size + tol_large > purchased_size + fit_offset) and (
        true_size + tol_small < purchased_size + fit_offset
    ):
        return "fit"
    return "not applicable"


def generate_transactions(consumer_features: pd.DataFrame, product_features: pd.DataFrame) -> pd.DataFrame:
    """Generate the transactions table linking consumers to products over time.

    For each consumer:
      - Sample a number of purchases and their dates,
      - Choose products available in the consumer’s country,
      - Sample a purchased size and compute the fit outcome.

    Args:
        consumer_features: DataFrame from generate_consumers.
        product_features: DataFrame from generate_products.

    Returns:
        Transactions DataFrame with columns:
            consumer_id, product_id, purchased_size, fit_outcome, transaction_date
    """
    transactions = []

    for _, consumer in consumer_features.iterrows():
        num_purchases = generate_purchases()
        purchase_dates = generate_sequential_purchases(consumer["start_date"], num_purchases)

        # Products available in this consumer's country
        available = product_features[
            product_features["available_countries"].apply(lambda xs: consumer["country"] in xs)
        ]

        for purchase_date in purchase_dates:
            product = available.sample(1).iloc[0]
            product_type = product["product_type"]

            purchased_size = sample_purchased_size(consumer, product_type)
            fit_outcome = calculate_fit(consumer, product, product_type, purchased_size)

            transactions.append(
                {
                    "consumer_id": consumer["consumer_id"],
                    "product_id": product["product_id"],
                    "purchased_size": purchased_size,
                    "fit_outcome": fit_outcome,
                    "transaction_date": purchase_date,
                }
            )

    transaction_df = pd.DataFrame(transactions)
    return transaction_df

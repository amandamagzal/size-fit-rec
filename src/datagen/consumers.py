"""Consumer data generation utilities.

Functions:
- assign_sizes(row): correlated upper/lower clothing sizes and shoe size
- assign_personal_margins(df): per-consumer tolerance margins
- generate_consumers(n): full consumer table with sizes, margins, and start_date
"""

import numpy as np
import pandas as pd

from datagen.constants import (
    GENDERS,
    COUNTRIES,
    AGE_MEAN,
    AGE_STD,
    SIZES,
    SIZE_DISTS,
    SHOE_SIZES,
    SHOE_SIZE_DIST,
    PERSONAL_MARGIN_BASE_SMALL_RANGE,
    PERSONAL_MARGIN_BASE_LARGE_RANGE,
    GENDER_MARGIN_ADJ_SMALL,
    GENDER_MARGIN_ADJ_LARGE,
    COUNTRY_MARGIN_ADJ_SMALL,
    COUNTRY_MARGIN_ADJ_LARGE,
    MARGINS_ROUND_DECIMALS,
)


def assign_sizes(row: pd.Series) -> pd.Series:
    """Assign correlated upper/lower clothing sizes and a shoe size.

    Clothing sizes are drawn from the gender-specific probability distribution.
    Shoe sizes use the gender-specific frequency distribution modulated by a
    Gaussian centered around a mean derived from the clothing size index.

    Args:
        row: A pandas Series with at least a 'gender' field.

    Returns:
        A pandas Series: [upper_size (str), lower_size (str), shoe_size (float)].
    """
    # Upper size index from gender distribution
    upper_size_idx = np.random.choice(range(len(SIZES)), p = SIZE_DISTS[row["gender"]])

    # Lower size index: usually same as upper, sometimes ±1
    if np.random.rand() < 0.80:
        lower_size_idx = upper_size_idx
    else:
        lower_size_idx = np.random.choice(
            [max(0, upper_size_idx - 1), min(len(SIZES) - 1, upper_size_idx + 1)]
        )

    # Base shoe-size frequency by gender
    base_prob = SHOE_SIZE_DIST[row["gender"]]

    # Map clothing size index → base mean shoe size via linear interpolation
    min_shoe, max_shoe = SHOE_SIZES[0], SHOE_SIZES[-1]
    base_mean_shoe = min_shoe + (max_shoe - min_shoe) * (upper_size_idx / (len(SIZES) - 1))

    # Gender offset for mean shoe size
    if row["gender"] == "Female":
        gender_offset = -0.5
    elif row["gender"] == "Male":
        gender_offset = 0.0
    else:
        gender_offset = -0.25
    mean_shoe = base_mean_shoe + gender_offset

    # Gaussian weights centered at mean_shoe (sigma small to keep it peaked)
    sigma = 0.5
    gaussian_weights = np.exp(-0.5 * ((SHOE_SIZES - mean_shoe) / sigma) ** 2)
    gaussian_weights = gaussian_weights / gaussian_weights.sum()

    # Combine base frequencies with Gaussian emphasis and sample
    combined_prob = base_prob * gaussian_weights
    combined_prob = combined_prob / combined_prob.sum()
    shoe_choice = np.random.choice(SHOE_SIZES, p=combined_prob)

    return pd.Series([SIZES[upper_size_idx], SIZES[lower_size_idx], shoe_choice])


def assign_personal_margins(consumer_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Generate personal tolerance margins for products that run small/large.

    Margins are sampled from a baseline and adjusted by gender and country.

    Args:
        consumer_df: DataFrame with 'gender' and 'country' columns.

    Returns:
        Tuple of two Series:
            - tolerance for "too small" (negative / tighter)
            - tolerance for "too large" (positive / looser)
    """
    n = len(consumer_df)

    # Baseline draws
    lo_s, hi_s = PERSONAL_MARGIN_BASE_SMALL_RANGE
    lo_l, hi_l = PERSONAL_MARGIN_BASE_LARGE_RANGE
    baseline_small = np.random.uniform(lo_s, hi_s, size=n)
    baseline_large = np.random.uniform(lo_l, hi_l, size=n)

    margins_small, margins_large = [], []

    # Enumerate to avoid relying on DataFrame index alignment
    for i, row in enumerate(consumer_df.itertuples(index=False)):
        g, c = row.gender, row.country

        m_small = baseline_small[i] + GENDER_MARGIN_ADJ_SMALL.get(g, 0.0) + COUNTRY_MARGIN_ADJ_SMALL.get(c, 0.0)
        m_large = baseline_large[i] + GENDER_MARGIN_ADJ_LARGE.get(g, 0.0) + COUNTRY_MARGIN_ADJ_LARGE.get(c, 0.0)

        margins_small.append(round(m_small, MARGINS_ROUND_DECIMALS))
        margins_large.append(round(m_large, MARGINS_ROUND_DECIMALS))

    return pd.Series(margins_small), pd.Series(margins_large)


def generate_consumers(n_consumers: int) -> pd.DataFrame:
    """Create the consumer table with demographics, sizes, margins, and start date.

    Args:
        n_consumers: Number of consumers to generate.

    Returns:
        DataFrame with columns:
            - consumer_id (str), gender (str), country (str), age (int)
            - upper_size (str), lower_size (str), shoe_size (float)
            - consumer_tolerance_too_small (float), consumer_tolerance_too_large (float)
            - start_date (datetime64[ns])
    """
    # Demographics
    consumer_features = pd.DataFrame(
        {
            "consumer_id": [f"c_{i}" for i in range(1, n_consumers + 1)],
            "gender": np.random.choice(GENDERS, size = n_consumers, p = [0.4, 0.45, 0.15]),
            "country": np.random.choice(COUNTRIES, size = n_consumers),
            "age": np.clip(np.random.normal(AGE_MEAN, AGE_STD, size = n_consumers), 18, 65).astype(int),
        }
    )

    # Sizes
    consumer_features[["upper_size", "lower_size", "shoe_size"]] = consumer_features.apply(
        assign_sizes, axis=1
    )

    # Personal margins
    (
        consumer_features["consumer_tolerance_too_small"],
        consumer_features["consumer_tolerance_too_large"],
    ) = assign_personal_margins(consumer_features)

    # First purchase date
    consumer_features["start_date"] = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, size = len(consumer_features)), unit = "D"
    )

    return consumer_features


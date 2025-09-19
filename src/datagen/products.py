"""Product data generation utilities.

Functions:
- assign_countries(): sample 1–3 available countries per product
- select_material(product_type): pick a material based on type-specific probabilities
- assign_fit_offset(row): compute fit offset from fit type, size accuracy, material, and section
- generate_products(n_products): build the product catalog dataframe
"""

import numpy as np
import pandas as pd

from datagen.constants import (
    COUNTRIES,
    GENDERS,
    PRODUCT_TYPES,
    FIT_TYPES,
    SIZE_ACCURACY,
    MATERIALS_BY_TYPE,
    FIT_TYPE_OFFSET_RANGES,
    SIZE_ACCURACY_OFFSET_RANGES,
    MATERIAL_FIT_INFLUENCE,
    GENDER_FIT_ADJUSTMENTS,
)


def assign_countries() -> list[str]:
    """Assign 1–3 countries where the product is available.

    Returns:
        A list of country names sampled without replacement from COUNTRIES.
    """
    num_countries = np.random.choice([1, 2, 3], p = [0.5, 0.3, 0.2])
    available_countries = list(np.random.choice(COUNTRIES, size = num_countries, replace = False))
    return available_countries


def select_material(product_type: str) -> str:
    """Select a material for a given product type using predefined probabilities.

    Args:
        product_type: One of PRODUCT_TYPES.

    Returns:
        The selected material name.
    """
    materials, probs = MATERIALS_BY_TYPE[product_type]
    selected_material = np.random.choice(materials, p=probs)
    return selected_material


def assign_fit_offset(row: pd.Series) -> float:
    """Assign a fit offset based on fit type, size accuracy, material, and section."""
    # Sample within configured ranges (adds natural per-item variability)
    lo, hi = FIT_TYPE_OFFSET_RANGES[row["fit_type"]]
    fit_offset = np.random.uniform(lo, hi)

    lo, hi = SIZE_ACCURACY_OFFSET_RANGES[row["size_accuracy"]]
    accuracy_offset = np.random.uniform(lo, hi)

    material_influence = MATERIAL_FIT_INFLUENCE.get(row["material"], 0.0)
    section_offset = GENDER_FIT_ADJUSTMENTS.get(row["section"], 0.0)

    total_offset = fit_offset + accuracy_offset + material_influence + section_offset
    return round(total_offset, 2)


def generate_products(n_products: int) -> pd.DataFrame:
    """Generate the product catalog with attributes and fit offsets.

    Args:
        n_products: Number of products to generate.

    Returns:
        DataFrame with columns:
            - product_id (str), section (str), product_type (str)
            - fit_type (str), size_accuracy (str), available_countries (list[str])
            - material (str), fit_offset (float)
    """
    product_features = pd.DataFrame(
        {
            "product_id": [f"a_{i}" for i in range(1, n_products + 1)],
            "section": np.random.choice(GENDERS, size = n_products),
            "product_type": np.random.choice(PRODUCT_TYPES, size = n_products),
            "fit_type": np.random.choice(FIT_TYPES, size = n_products, p = [0.3, 0.4, 0.3]),
            "size_accuracy": np.random.choice(SIZE_ACCURACY, size = n_products, p = [0.1, 0.8, 0.1]),
        }
    )

    # Availability
    product_features["available_countries"] = product_features.apply(
        lambda _: assign_countries(), axis=1
    )

    # Material per product type
    product_features["material"] = product_features["product_type"].apply(select_material)

    # Fit offset
    product_features["fit_offset"] = product_features.apply(assign_fit_offset, axis=1)

    return product_features

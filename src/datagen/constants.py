"""Global constants for synthetic data generation.

This module defines:
- Consumer/global enums (genders, countries, size scales)
- Default dataset sizes
- Distributions for clothing and shoe sizes (by gender)
- Product catalog knobs (types, materials, fit labels)
"""

import numpy as np

# -----------------------------
# Global enums / categorical sets
# -----------------------------
GENDERS = ["Male", "Female", "Other"]
COUNTRIES = ["USA", "Germany", "UK", "Japan"]


# -----------------------------
# Consumers
# -----------------------------
# Default number of consumers to generate (can be overridden by callers)
N_CONSUMERS = 1000

# Age distribution (years)
AGE_MEAN = 35
AGE_STD = 10

# Universal clothing sizes (ordinal, smallest → largest)
SIZES = ["2XS", "XS", "S", "M", "L", "XL", "2XL"]

# Clothing size distributions by gender (must sum to 1.0)
SIZE_DISTS = {
    "Female": np.array([1, 5, 30, 26, 20, 10, 8]) / 100.0,
    "Male":   np.array([0.5, 1, 7, 28, 30, 20, 13.5]) / 100.0,
}
# "Other" as the mean of Female and Male distributions
SIZE_DISTS["Other"] = (SIZE_DISTS["Female"] + SIZE_DISTS["Male"]) / 2.0

# Shoe sizes (UK) from 4.5 to 15.5 in 0.5 increments
SHOE_SIZES = np.arange(4.5, 16, 0.5)

# Empirical-ish shoe size frequencies (arbitrary scale; will be normalized)
FREQUENCY_MALE = np.array(
    [1, 2, 3, 4, 6, 7, 9, 11, 12, 14, 12, 10, 9, 7, 6, 4, 3, 2, 2, 1, 1, 0, 0]
)
FREQUENCY_FEMALE = np.array(
    [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 14, 12, 10, 8, 7, 5, 4, 3, 2, 1, 1, 0, 0]
)

# Normalized shoe-size probabilities by gender; "Other" is the mean of the two
SHOE_SIZE_DIST = {
    "Female": FREQUENCY_FEMALE / FREQUENCY_FEMALE.sum(),
    "Male":   FREQUENCY_MALE   / FREQUENCY_MALE.sum(),
}
SHOE_SIZE_DIST["Other"] = (SHOE_SIZE_DIST["Female"] + SHOE_SIZE_DIST["Male"]) / 2.0

# Personal margin baseline ranges (sampled per consumer)
PERSONAL_MARGIN_BASE_SMALL_RANGE = (-0.5, 0.0)   # tolerance for "too small"
PERSONAL_MARGIN_BASE_LARGE_RANGE = ( 0.0, 0.65)  # tolerance for "too large"

# Deterministic adjustments by gender and country
GENDER_MARGIN_ADJ_SMALL = {"Male": -0.1, "Female": 0.1, "Other": 0.0}
GENDER_MARGIN_ADJ_LARGE = {"Male":  0.1, "Female": -0.1, "Other": 0.0}

COUNTRY_MARGIN_ADJ_SMALL = {"USA": -0.1, "Germany": 0.0, "UK": 0.1, "Japan": -0.05}
COUNTRY_MARGIN_ADJ_LARGE = {"USA":  0.1, "Germany": 0.0, "UK": -0.1, "Japan":  0.05}

# Rounding precision for margins
MARGINS_ROUND_DECIMALS = 2


# -----------------------------
# Products
# -----------------------------
# Default number of unique products to generate (can be overridden)
N_PRODUCTS = 1000

# Product taxonomy
PRODUCT_TYPES = ["T-Shirts & Tops", "Sweatshirts/Hoodies", "Sportswear",
                 "Pants & Leggings", "Jackets", "Shoes"]

# Materials and their probabilities per product type (probs must sum to 1.0)
MATERIALS_BY_TYPE = {
    "T-Shirts & Tops":     (["Cotton", "Recycled Polyester", "Polyester"], [0.5, 0.3, 0.2]),
    "Sweatshirts/Hoodies": (["Cotton", "Recycled Polyester", "Polyester"], [0.6, 0.3, 0.1]),
    "Sportswear":          (["Aeroready Technology", "Recycled Polyester", "Spandex"], [0.5, 0.3, 0.2]),
    "Pants & Leggings":    (["Polyester", "Spandex", "Recycled Polyester", "Nylon"], [0.5, 0.2, 0.2, 0.1]),
    "Jackets":             (["Polyester", "Recycled Polyester", "Nylon", "Primeknit"], [0.5, 0.3, 0.1, 0.1]),
    "Shoes":               (["Primeknit", "Recycled Polyester", "Parley Ocean Plastic"], [0.5, 0.3, 0.2]),
}

# Fit offset ranges (sampled per product)
FIT_TYPE_OFFSET_RANGES = {
    "slim":    (-0.6, -0.4),
    "regular": (-0.2,  0.2),
    "loose":   ( 0.4,  0.6),
}

SIZE_ACCURACY_OFFSET_RANGES = {
    "runs small":  (-0.6, -0.4),
    "true to size":(-0.2,  0.2),
    "runs large":  ( 0.4,  0.6),
}

# Deterministic adjustments
MATERIAL_FIT_INFLUENCE = {
    "Cotton": -0.1,
    "Recycled Polyester": -0.05,
    "Polyester": 0.0,
    "Spandex": 0.2,
    "Aeroready Technology": 0.1,
    "Nylon": 0.15,
    "Primeknit": 0.2,
    "Parley Ocean Plastic": 0.1,
}

GENDER_FIT_ADJUSTMENTS = {
    "Male": 0.1,
    "Female": -0.1,
    "Other": 0.0,
}

# Fit labels
FIT_TYPES = ["slim", "regular", "loose"]
SIZE_ACCURACY = ["runs small", "true to size", "runs large"]

import pandas as pd
import numpy as np
from datagen.constants import *


def assign_countries():

    """
    Assigns each product to 1-3 randomly selected countries where it is available.

    Returns:
    - A list of randomly selected country names from the predefined 'countries' list.
    """
    
    # Select the number of countries for an product
    num_countries = np.random.choice([1, 2, 3], p = [0.5, 0.3, 0.2])
    
    # Select random countries
    available_countries = list(np.random.choice(countries, size = num_countries, replace = False))
    
    return available_countries


def select_material(product_type):

    """
    Selects a material for a given product type based on predefined probabilities.
    
    Parameters:
    - product_type: Type of the product for which material needs to be selected.
    
    Returns:
    - A selected material as a string.
    """

    # Extract materials and probabilities for the specific product type
    materials, probs = materials_by_type[product_type]
    
    # Select a random material
    selected_material = np.random.choice(materials, p = probs)
    
    return selected_material


# Function to assign a variable fit offset based on both fit type and size accuracy
def assign_fit_offset(row):

    """
    Assigns a fit offset to a product based on its fit type, size accuracy, material, country and gender.
    
    Parameters:
    - row: DataFrame row containing product information.
    
    Returns:
    - A numerical fit offset that affects how the product fits.
    """
    
    # Define adjustments by fit type with random variability
    fit_type_offsets = {
        'slim': np.random.uniform(-0.6, -0.4),  # Slim fits tend to be tighter
        'regular': np.random.uniform(-0.2, 0.2),  # Regular fits are around standard
        'loose': np.random.uniform(0.4, 0.6)   # Loose fits offer more room
    }
    fit_offset = fit_type_offsets[row['fit_type']]

    # Define adjustments by size accuracy with random variability
    size_accuracy_offsets = {
        'runs small': np.random.uniform(-0.6, -0.4),  # Size smaller than expected
        'true to size': np.random.uniform(-0.2, 0.2),   # Size as expected
        'runs large': np.random.uniform(0.4, 0.6)    # Size larger than expected
    }
    acccuracy_offset = size_accuracy_offsets[row['size_accuracy']]

    # Define material influence on fit
    material_fit_influence = {
        'Cotton': -0.1,
        'Recycled Polyester': -0.05,
        'Polyester': 0,
        'Spandex': 0.2,
        'Aeroready Technology': 0.1,
        'Nylon': 0.15,
        'Primeknit': 0.2,
        'Parley Ocean Plastic': 0.1
    }
    material_influence = material_fit_influence.get(row['material'], 0)

    # Define gender adjustment
    gender_adjustments = {
        "Male": 0.1,  # Slightly larger fit for males
        "Female": -0.1,  # Slightly smaller fit for females
        "Other": 0.0,  # No adjustment for other genders
    }
    section_offset = gender_adjustments[row['section']]

    # Calculate total offset by combining all dimensions
    total_offset = fit_offset + acccuracy_offset + material_influence + section_offset
    
    return round(total_offset, 2)


def generate_products(n_products):

    # Generate product features
    product_features = pd.DataFrame({
        'product_id': [f'a_{i}' for i in range(1, n_products + 1)],
        'section': np.random.choice(genders, size = n_products),
        'product_type': np.random.choice(product_types, size = n_products),
        'fit_type': np.random.choice(fit_types, size = n_products, p =[ 0.3, 0.4, 0.3]),
        'size_accuracy': np.random.choice(size_accuracy, size = n_products, p = [0.1, 0.8, 0.1])
    })

    # Assign each product to multiple countries
    product_features['available_countries'] = product_features.apply(lambda _: assign_countries(), axis=1)

    # Assign material to each product based on its type
    product_features['material'] = product_features['product_type'].apply(select_material)

    # Update product features with combined fit_offset
    product_features['fit_offset'] = product_features.apply(assign_fit_offset, axis = 1)

    return product_features

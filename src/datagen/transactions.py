import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import truncnorm

from datagen.constants import *


def generate_purchases(max_purchases = 3000):
    
    """
    Generates a realistic number of purchases for a consumer based on a truncated normal distribution.
    
    Parameters:
    - max_purchases: The maximum number of purchases allowed.
    
    Returns:
    - An integer representing the number of purchases.
    """
    
    mean = 5
    std_dev = 2.0
    lower, upper = 1, max_purchases
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    base_sample = truncnorm(a, b, loc = mean, scale = std_dev).rvs()
    purchases = int(round(base_sample))
    
    # Ensure the number is within the valid range
    purchases = max(lower, min(purchases, upper))

    # Adjust purchase number occasionally
    if purchases != 1 and np.random.rand() < 0.1:
        purchases = 1
        
    return purchases


def generate_sequential_purchases(start_date, num_purchases):

    """
    Generates sequential purchase dates for a given number of purchases starting from a specified date.
    
    Parameters:
    - start_date: The date of the first purchase.
    - num_purchases: The total number of purchases to generate dates for.
    
    Returns:
    - A list of datetime objects representing the dates of each purchase.
    """
    
    purchase_dates = []
    
    for _ in range(num_purchases):
        
        # Days between purchases
        days_between_purchases = np.random.randint(1, 90)  # Randomized days between purchases
        
        # Calculate purchase date
        purchase_date = start_date + timedelta(days = days_between_purchases)
        purchase_dates.append(purchase_date)
        
        # Update start date for the next purchase
        start_date = purchase_date
        
    return purchase_dates


def sample_purchased_size(consumer, product_type):
    
    """
    Samples a purchased size around the consumer's true size with a realistic distribution based on the product type.
    
    Parameters:
    - consumer: The consumer data row.
    - product_type: The type of product being purchased.
    
    Returns:
    - A string or number representing the purchased size.
    """
    
    # Determine the size type to use based on the product type
    if 'Shoes' in product_type:
        size_list = shoe_sizes
        true_size = consumer['shoe_size']
    elif 'Pants & Leggings' in product_type:
        size_list = sizes
        true_size = consumer['lower_size']
    else:
        size_list = sizes
        true_size = consumer['upper_size']
    
    # Get the current index of the true size within the size list
    current_index = np.where(np.array(size_list) == true_size)[0][0]
    
    # Choose a size variation
    size_variation = np.random.choice([-1, 0, 1], p = [0.15, 0.7, 0.15])  # Mostly buys true size, sometimes one size off
    
    # Calculate the new index
    new_index = current_index + size_variation
    
    # Make sure the size is within a valid range
    if new_index > (len(size_list) - 1):
        return size_list[len(size_list) - 1]
    elif new_index < 0:
        return size_list[0]
    else:
        return size_list[new_index]


# Function to simulate a transaction and calculate fit outcome
def calculate_fit(consumer, product, product_type, purchased_size):
    
    """
    Calculates the fit outcome based on consumer tolerance and product fit offset.
    
    Parameters:
    - consumer: The consumer data row.
    - product_type: The type of product being evaluated.
    - purchased_size: The size that was purchased by the consumer.
    
    Returns:
    - A string indicating the fit outcome ('too small', 'too large', 'fit', 'not applicable').
    """

    size_mapping = {'2XS': 1, 'XS': 2, 'S': 3, 'M': 4, 'L': 5, 'XL': 6, '2XL': 7}
    
    # Determine the size type to use based on the product type
    if 'Shoes' in product_type:
        true_size = consumer['shoe_size']
    elif 'Pants & Leggings' in product_type:
        size_list = size_mapping
        true_size = size_mapping[consumer['lower_size']]
    else:
        size_list = size_mapping
        true_size = size_mapping[consumer['upper_size']]

    # Convert purchased size to numerical value if it's a string
    if isinstance(purchased_size, str):
        purchased_size = size_mapping[purchased_size]

    # Fit offset and tolerances as numerical adjustments
    tolerance_too_small = consumer['consumer_tolerance_too_small']
    tolerance_too_large = consumer['consumer_tolerance_too_large']
    fit_offset = product['fit_offset']
    
    # Calculate the adjusted true size with the fit offset
    adjusted_size = true_size + fit_offset

    # Determine fit outcome based on numerical comparisons
    if purchased_size < adjusted_size + tolerance_too_small:
        return "too small"
    elif purchased_size > adjusted_size + tolerance_too_large:
        return "too large"
    elif (true_size +  tolerance_too_large > purchased_size + fit_offset) & (true_size + tolerance_too_small < purchased_size + fit_offset):
        return "fit"
    return "not applicable"


def generate_transactions(consumer_features, product_features):

    # Generate transactions
    transactions = []
    for idx, consumer in consumer_features.iterrows():

        # Get the customer's number of purchases and their dates
        num_purchases = generate_purchases()
        purchase_dates = generate_sequential_purchases(consumer['start_date'], num_purchases)

        # Iterate through the purchases
        for purchase_date in purchase_dates:

            # Filter products available in the consumer's country
            available_products = product_features[product_features['available_countries'].apply(lambda x: consumer['country'] in x)]
            
            # Randomly select an product
            product = available_products.sample(1).iloc[0]
            product_type = product['product_type']

            # Get the pourchased size and fit outcome
            purchased_size = sample_purchased_size(consumer, product_type)
            fit_outcome = calculate_fit(consumer, product, product_type, purchased_size)
            
            transactions.append({
                'consumer_id': consumer['consumer_id'],
                'product_id': product['product_id'],
                'purchased_size': purchased_size,
                'fit_outcome': fit_outcome,
                'transaction_date': purchase_date
            })

    # Create DataFrame from transactions
    transaction_df = pd.DataFrame(transactions)

    return transaction_df

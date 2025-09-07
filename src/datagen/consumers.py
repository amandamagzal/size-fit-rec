import pandas as pd
import numpy as np
from datagen.constants import *


# Function to generate correlated upper and lower sizes based on gender-specific distributions
def assign_sizes(row):
    
    """
    Assigns correlated upper and lower clothing sizes and a shoe size to a consumer.
    
    For clothing sizes, it randomly selects an index using a gender-specific probability distribution.
    For shoe sizes, it uses the gender-specific frequency distribution combined with a Gaussian 
    centered around a mean shoe size derived from the clothing size index.
    
    Parameters:
    - row: DataFrame row containing consumer data.
    
    Returns:
    - Series containing upper_size, lower_size, and shoe_size.
    """

    # Assign upper clothing parts size
    upper_size_idx = np.random.choice(range(len(sizes)), p = size_dists[row['gender']])
    
    # Assign lower clothing part size by applying a simple correlation factor: mostly the same size, sometimes one size apart
    if np.random.rand() < 0.80:
        lower_size_idx = upper_size_idx
    else:
        lower_size_idx = np.random.choice([max(0, upper_size_idx - 1), min(len(sizes) - 1, upper_size_idx + 1)])

    # Assign shoe size
    shoe_choice = np.random.choice(shoe_sizes, p = shoe_size_dist[row['gender']])

    # Get the base frequency distribution for shoe sizes based on gender.
    if row['gender'] == 'Female':
        base_freq = frequency_female
    elif row['gender'] == 'Male':
        base_freq = frequency_male
    else:
        base_freq = (frequency_female + frequency_male) / 2.0

    # Normalize the frequency distribution to form probabilities.
    base_prob = base_freq / base_freq.sum()
    
    # Map the clothing size index (upper_size_idx) to a base mean shoe size using linear interpolation.
    # The mapping is from the clothing size range (indices 0 to len(sizes)-1) to the shoe size range.
    min_shoe = shoe_sizes[0]
    max_shoe = shoe_sizes[-1]
    base_mean_shoe = min_shoe + (max_shoe - min_shoe) * (upper_size_idx / (len(sizes) - 1))
    
    # Apply a gender-specific offset to adjust the mean shoe size:
    # For females, subtract 0.5; for males, no offset; for "Other", use an intermediate offset.
    if row['gender'] == 'Female':
        gender_offset = -0.5
    elif row['gender'] == 'Male':
        gender_offset = 0.0
    else:
        gender_offset = -0.25
    mean_shoe = base_mean_shoe + gender_offset
    
    # Define a Gaussian (normal) distribution over the shoe sizes centered at mean_shoe.
    # A small sigma (e.g., 0.5) ensures the distribution is peaked around the mean.
    sigma = 0.5
    gaussian_weights = np.exp(-0.5 * ((shoe_sizes - mean_shoe) / sigma)**2)
    gaussian_weights = gaussian_weights / gaussian_weights.sum()  # Normalize to sum to 1
    
    # Combine the base frequency probabilities with the Gaussian weights.
    # This product emphasizes shoe sizes near the mean while retaining the overall frequency distribution.
    combined_prob = base_prob * gaussian_weights
    combined_prob = combined_prob / combined_prob.sum()  # Renormalize to sum to 1
    
    # Sample a shoe size from the combined probability distribution.
    shoe_choice = np.random.choice(shoe_sizes, p=combined_prob)

    return pd.Series([sizes[upper_size_idx], sizes[lower_size_idx], shoe_choice])


def assign_personal_margins(consumer_df):
    
    """
    Generates personal size tolerance margins for consumers, adjusted by gender and country.
    
    Parameters:
    - consumer_df: DataFrame containing consumer data with 'gender' and 'country' columns.
    
    Returns:
    - Two pandas Series representing the tolerance margins for products that run too small and too large.
    """
    
    n = len(consumer_df)
    
    # Baseline margins (sampled uniformly)
    baseline_small = np.random.uniform(-0.5, 0, size = n)   # Baseline tolerance for "too small"
    baseline_large = np.random.uniform(0, 0.65, size = n)     # Baseline tolerance for "too large"
    
    # Define gender-based adjustments for margins
    gender_adj_small = {'Male': -0.1, 'Female': 0.1, 'Other': 0.0}
    gender_adj_large = {'Male': 0.1, 'Female': -0.1, 'Other': 0.0}
    
    # Define country-based adjustments for margins
    country_adj_small = {'USA': -0.1, 'Germany': 0.0, 'UK': 0.1, 'Japan': -0.05}
    country_adj_large = {'USA': 0.1, 'Germany': 0.0, 'UK': -0.1, 'Japan': 0.05}
    
    margins_small = []
    margins_large = []
    
    # Loop over each consumer and adjust the baseline margins using gender and country
    for i, row in consumer_df.iterrows():
        
        g = row['gender']
        c = row['country']
        
        # Adjust baseline with gender and country corrections
        margin_small = baseline_small[i] + gender_adj_small.get(g, 0) + country_adj_small.get(c, 0)
        margin_large = baseline_large[i] + gender_adj_large.get(g, 0) + country_adj_large.get(c, 0)
        
        margins_small.append(round(margin_small, 2))
        margins_large.append(round(margin_large, 2))
    
    return pd.Series(margins_small), pd.Series(margins_large)



def generate_consumers(n_consumers):
    
    # Generate consumer features
    consumer_features = pd.DataFrame({
        'consumer_id': [f'c_{i}' for i in range(1, n_consumers + 1)],
        'gender': np.random.choice(genders, size = n_consumers, p = [0.4, 0.45, 0.15]),
        'country': np.random.choice(countries, size = n_consumers),
        'age': np.clip(np.random.normal(age_mean, age_std, size = n_consumers), 18, 65).astype(int)
    })

    # Assign sizes
    consumer_features[['upper_size', 'lower_size', 'shoe_size']] = consumer_features.apply(assign_sizes, axis = 1)

    # Add personal margins to consumers
    consumer_features['consumer_tolerance_too_small'], consumer_features['consumer_tolerance_too_large'] = assign_personal_margins(consumer_features)

    # Add date of first purchase
    consumer_features['start_date'] = pd.to_datetime('2021-01-01') + pd.to_timedelta(np.random.randint(0, 365, size = len(consumer_features)), unit = 'd')

    return consumer_features

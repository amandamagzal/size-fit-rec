import numpy as np


genders = ['Male', 'Female', 'Other']
countries = ['USA', 'Germany', 'UK', 'Japan']



### CONSUMERS ###


# Total number of customers
n_consumers = 1000

# Average age and standard deviation
age_mean = 35
age_std = 10


# Define size distributions by gender (the gender "Other" is simply the average of male and female)

# Universal clothing sizes for simplicity
sizes = ['2XS', 'XS', 'S', 'M', 'L', 'XL', '2XL']
size_dists = {
    'Female': np.array([1, 5, 30, 26, 20, 10, 8]) / 100,
    'Male': np.array([0.5, 1, 7, 28, 30, 20, 13.5]) / 100
}
size_dists['Other'] = (size_dists['Female'] + size_dists['Male']) / 2

# Shoe sizes from 4.5 to 15.5 (UK scale for simplicity)
shoe_sizes = np.arange(4.5, 16, 0.5)

# Shoe sizes frequency extracted from an online resource
frequency_male = np.array([1, 2, 3, 4, 6, 7, 9, 11, 12, 14, 12, 10, 9, 7, 6, 4, 3, 2, 2, 1, 1, 0, 0])
frequency_female = np.array([0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 14, 12, 10, 8, 7, 5, 4, 3, 2, 1, 1, 0, 0])

# Normalize to calculate probabilites
shoe_size_dist = {
    'Female': frequency_female / frequency_female.sum(),
    'Male': frequency_male / frequency_male.sum()
}
shoe_size_dist['Other'] = (shoe_size_dist['Female'] + shoe_size_dist['Male']) / 2



### PRODUCTS ###


# Total number of unique products
n_products = 1000  


# Define product types and associated materials
product_types = ['T-Shirts & Tops', 'Sweatshirts/Hoodies', 'Sportswear', 'Pants & Leggings', 'Jackets', 'Shoes']
materials_by_type = {
    'T-Shirts & Tops': (['Cotton', 'Recycled Polyester', 'Polyester'], [0.5, 0.3, 0.2]),
    'Sweatshirts/Hoodies': (['Cotton', 'Recycled Polyester', 'Polyester'], [0.6, 0.3, 0.1]),
    'Sportswear': (['Aeroready Technology', 'Recycled Polyester', 'Spandex'], [0.5, 0.3, 0.2]),
    'Pants & Leggings': (['Polyester', 'Spandex', 'Recycled Polyester', 'Nylon'], [0.5, 0.2, 0.2, 0.1]),
    'Jackets': (['Polyester', 'Recycled Polyester', 'Nylon', 'Primeknit'], [0.5, 0.3, 0.1, 0.1]),
    'Shoes': (['Primeknit', 'Recycled Polyester', 'Parley Ocean Plastic'], [0.5, 0.3, 0.2])
}


# Define fit types and sizing accuracy
fit_types = ['slim', 'regular', 'loose']
size_accuracy = ['runs small', 'true to size', 'runs large']



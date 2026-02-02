import random
import numpy as np

def augment_features(data, is_training=True):
    """
    Augments property data with price-aware synthetic features: amenitiesScore and propertyAge.
    
    Logic:
    - If 'price' is present:
        - amenitiesScore: Positively correlated with price (higher price -> higher score).
        - propertyAge: Negatively correlated with price (higher price -> newer property).
    - If 'price' is missing (prediction time):
        - Fallback to reasonable defaults or logic based on other features like 'size'.
    
    Args:
        data (dict or list): A single property dictionary or a list of property dictionaries.
        is_training (bool): Whether we are in training mode (where price is always known).
        
    Returns:
        The augmented data.
    """
    if isinstance(data, list):
        for item in data:
            _augment_single(item)
        return data
    else:
        return _augment_single(data)

def _augment_single(item):
    """Internal helper for single item augmentation with price-aware logic."""
    price = item.get('price')
    size = item.get('size', 1000)
    
    # Base logic for amenitiesScore (5-9 range)
    if 'amenitiesScore' not in item or item['amenitiesScore'] is None:
        if price:
            # Scale score based on price (Log scale to prevent extreme outliers)
            # Assume 2M to 20M range for scaling
            # Log10(2,000,000) approx 6.3, Log10(20,000,000) approx 7.3
            normalized_price = np.clip((np.log10(price) - 6.3) / (7.3 - 6.3), 0, 1)
            item['amenitiesScore'] = round(5.0 + (normalized_price * 4.0) + random.uniform(-0.3, 0.3), 1)
        else:
            # Fallback for prediction: use size as a proxy for premiumness
            normalized_size = np.clip((size - 500) / (3000 - 500), 0, 1)
            item['amenitiesScore'] = round(5.0 + (normalized_size * 3.0) + random.uniform(-0.3, 0.3), 1)
        
        # Ensure bounds
        item['amenitiesScore'] = max(4.0, min(10.0, item['amenitiesScore']))

    # Base logic for propertyAge (0-30 range)
    if 'propertyAge' not in item or item['propertyAge'] is None:
        if price:
            # Negative correlation: Higher price -> Lower age (Newer)
            normalized_price = np.clip((np.log10(price) - 6.3) / (7.3 - 6.3), 0, 1)
            # Newer properties (0-5 years) for high price, older (15-30) for low price
            base_age = 20 - (normalized_price * 15)
            item['propertyAge'] = int(max(0, base_age + random.randint(-2, 2)))
        else:
            # Fallback for prediction
            item['propertyAge'] = random.randint(5, 15)
            
    return item

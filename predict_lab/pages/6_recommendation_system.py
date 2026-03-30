import os
import requests
import pandas as pd

API_KEY = "e480a9a869824f83b18091740020e3fa"
BASE_URL = "https://gamebrain.co/api/"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

try:
    response = requests.get(BASE_URL, headers=headers)
    response.raise_for_status()  # Raises error for bad status codes (4xx, 5xx)
    
    data = response.json()
    
    # Handle both list and dict responses
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Try to find the list of games inside the dict
        for key, value in data.items():
            if isinstance(value, list):
                df = pd.DataFrame(value)
                break
        else:
            df = pd.DataFrame([data])  # Wrap single dict in a list

    print(df.head())
    df.to_csv("games_dataset.csv", index=False)
    print("Saved to games_dataset.csv")

except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except requests.exceptions.ConnectionError:
    print("Connection error: Could not reach the API.")
except requests.exceptions.Timeout:
    print("Request timed out.")
except ValueError as e:
    print(f"Error parsing response: {e}")
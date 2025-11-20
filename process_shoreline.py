# process_shoreline_data.py (Updated Version)
# A simple script to convert the CSV data of buildings on filled land into a JSON format for the web dashboard.
# This version correctly handles missing values (NaN) by converting them to null for valid JSON output.

import pandas as pd
import numpy as np
import json


def convert_csv_to_json(csv_path, json_path):
    """
    Reads building data from a CSV file, handles missing values,
    and exports it as a valid JSON array.

    Args:
        csv_path (str): The path to the input CSV file.
        json_path (str): The path for the output JSON file.
    """
    print(f"Reading data from {csv_path}...")
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Ensure essential columns exist
    required_columns = ['LATITUDE', 'LONGITUDE', 'OCC_CLS', 'material_type', 'foundation_type', 'year_built']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
            print(f"Warning: Column '{col}' not found. Added as an empty column.")

    print(f"Loaded {len(df)} records.")

    # --- START: ADDED FIX ---
    # Replace all occurrences of pandas missing values (NA, nan) with Python's None.
    # Python's None will be correctly serialized to JSON's null.
    # This is the key change to fix the SyntaxError.
    df = df.replace({np.nan: None})
    # --- END: ADDED FIX ---

    # Convert the DataFrame to a list of dictionaries
    records = df.to_dict(orient='records')

    print(f"Writing valid JSON data to {json_path}...")
    # Write the list of dictionaries to a JSON file
    with open(json_path, 'w') as f:
        json.dump(records, f)

    print("Conversion complete!")


if __name__ == "__main__":
    # Define the input and output file paths
    input_csv = 'buildings_on_1630_filled_land.csv'
    output_json = 'historic_shoreline_buildings.json'

    # Run the conversion function
    convert_csv_to_json(input_csv, output_json)
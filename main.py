import pandas as pd
import dotenv, os

dotenv.load_dotenv()

DATASET_COLUMN_NAMES = {
    "X1": "Relative Compactness",
    "X2": "Surface Area",
    "X3": "Wall Area",
    "X4": "Roof Area",
    "X5": "Overall Height",
    "X6": "Orientation",
    "X7": "Glazing Area",
    "X8": "Glazing Area Distribution",
    "Y1": "Heating Load",
    "Y2": "Cooling Load"
}

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
if __name__ == "__main__":
    file_path = os.getenv("DATASET_PATH")
    data: pd.DataFrame = load_csv_data(file_path)
    data.rename(columns=DATASET_COLUMN_NAMES, inplace=True) 
    if data is not None:
        print(data.head()) 
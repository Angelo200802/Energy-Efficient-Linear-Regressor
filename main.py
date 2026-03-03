import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
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

def plot_distribution(data: pd.DataFrame, label: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[label], bins=30, kde=True)
    plt.title(f"Distribution of {label}")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

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
    plot_correlation_matrix(data)
    plot_distribution(data, "Heating Load")
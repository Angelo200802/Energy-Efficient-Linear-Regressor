import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dotenv
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.linalg import det, cond
from scipy.stats import zscore

# Configurazione stile grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

def evaluate_multicollinearity(df):
    # 1. Matrice di Correlazione
    corr_matrix = df.corr()
    
    # 2. Determinante
    d = det(corr_matrix)
    
    # 3. Condition Number (sulla matrice dei dati standardizzata)
    c_number = cond(df.values)
    
    # 4. VIF e R^2 ausiliario
    # Aggiungiamo una costante per il calcolo del VIF (richiesto da statsmodels)
    vif_data = pd.DataFrame()
    vif_data["Variabile"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    # Calcolo R^2 ausiliario partendo dal VIF: R2 = 1 - (1/VIF)
    vif_data["R2_ausiliario"] = 1 - (1 / vif_data["VIF"])
    
    print(f"Determinante della matrice di corr: {d:.6f}")
    print(f"Condition Number: {c_number:.2f}")
    print("\nAnalisi VIF e R2:")
    print(vif_data)

def print_covariance_matrix(data: pd.DataFrame) -> None:
    """Stampa la matrice di varianze e covarianze."""
    cov_matrix = data.cov()
    print("\n=== Matrice di Varianze e Covarianze ===")
    print(cov_matrix)
    print(f"\nDimensione: {cov_matrix.shape}")
    print(f"\nDiagonale (varianze):\n{cov_matrix.values.diagonal()}")

def plot_distribution(data: pd.DataFrame, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[label], bins=30, kde=True, ax=ax, color='steelblue')
    ax.set_title(f"Distribution of {label}", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./img/{label.lower().replace(' ', '_')}_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", 
                cbar_kws={'label': 'Correlation'}, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix", fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("./img/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_covariance_matrix(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(data.cov(), annot=True, cmap='RdBu_r', fmt=".2f", center=0,
                cbar_kws={'label': 'Covariance'}, ax=ax, linewidths=0.5)
    ax.set_title("🌸 Covariance Matrix 🌸", fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("./img/covariance_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_regression_matrix(df, title="Matrice di dispersione con rette di regressione"):
    """
    Replica lo stile R-like: istogrammi grigi, punti neri e linee verdi.
    Risolto il TypeError per regplot.
    """
    sns.set_theme(style="ticks")
    
    # Inizializziamo la griglia
    grid = sns.PairGrid(df)
    
    # 1. Diagonale: Istogrammi grigi
    grid.map_diag(plt.hist, color='darkgrey', edgecolor='black', bins=10)
    
    # 2. Fuori diagonale: Scatter plot con retta VERDE
    # Nota: 'lw' e 'ls' ora sono dentro line_kws
    grid.map_offdiag(sns.regplot, 
                     scatter_kws={'color': 'black', 's': 10}, 
                     line_kws={'color': 'lawngreen', 'lw': 1.5, 'ls': '-'})

    # 3. Label variabili all'interno della diagonale
    for i, var in enumerate(df.columns):
        # Rimuove le etichette esterne per pulizia (come nello screenshot)
        grid.axes[i, 0].set_ylabel("")
        grid.axes[-1, i].set_xlabel("")
        
        # Scrive il nome della variabile al centro del plot diagonale
        grid.axes[i, i].annotate(var, xy=(0.5, 0.85), xycoords='axes fraction',
                                 ha='center', va='top', fontsize=12, 
                                 fontweight='normal')

    # Titolo superiore
    grid.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    # plt.savefig("./img/scatter_regression_matrix.png", dpi=300, bbox_inches='tight')
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
    
    print(data.head())
    print("Data Summary:")
    data.describe() #aggiungi mediana
    #plot_distribution(data, "Heating Load")
    #plot_distribution(data, "Cooling Load")
    #plot_covariance_matrix(data)
    #plot_correlation_matrix(data)
    #plot_regression_matrix(data)
    data_zscore = data.drop(columns=["Orientation"]).apply(zscore)
    evaluate_multicollinearity(data_zscore.drop(columns=["Heating Load", "Cooling Load","Relative Compactness", "Surface Area", "Roof Area"]))

    # 1. Creiamo le variabili dummy per l'orientamento (0 o 1)
    df_dummies = pd.get_dummies(data, columns=['Orientation'], prefix='Ori', drop_first=True)
    print(df_dummies.columns.tolist())
    
    # 2. Prepariamo la matrice X rimuovendo i target e le variabili ridondanti
    X = df_dummies.drop(columns=[
        "Heating Load", "Cooling Load", 
        "Relative Compactness", "Surface Area", "Roof Area"
    ])
    X = X.astype(float) # Convertiamo tutto in float

    # 3. STANDARDIZZAZIONE (Questo uccide il Condition Number gigante!)
    colonne_continue = ["Wall Area", "Overall Height", "Glazing Area", "Glazing Area Distribution"]
    X[colonne_continue] = X[colonne_continue].apply(zscore)

    # 4. FEATURE ENGINEERING (Le vere moltiplicazioni)
    X["GlazingxGlazingDist"] = X["Glazing Area"] * X["Glazing Area Distribution"]
    X["WallxHeight"] = X["Wall Area"] * X["Overall Height"]
    
 
    
    # Rimuoviamo gli orientamenti "puri" per non confondere il modello, 
    # teniamo solo le loro interazioni con il vetro
    X = X.drop(columns=[c for c in ["Ori_3", "Ori_4", "Ori_5"] if c in X.columns])

    # 5. Target e Modello
    y_heating = data["Heating Load"]
    y_cooling = data["Cooling Load"]

    X = sm.add_constant(X) # Aggiungiamo l'intercetta alla fine
    
    model_heating = sm.OLS(y_heating, X).fit()
    model_cooling = sm.OLS(y_cooling, X).fit()

    print(model_heating.summary())
    print("\n=== RISULTATI COOLING LOAD ===")
    print(model_cooling.summary())

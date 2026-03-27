import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dotenv
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from numpy.linalg import det, cond
from scipy.stats import zscore
import numpy as np
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

def transform_to_wls(ols_model):
    """
    Riceve un modello OLS di statsmodels e restituisce un modello WLS
    pesato per l'inverso del quadrato delle ordinate stimate (1 / y_hat^2).
    """
    # 1. Estrazione dei valori predetti (ordinate stimate)
    y_fitted = ols_model.fittedvalues
    
    # 2. Controllo stabilità numerica 
    # (I carichi termici Y1, Y2 sono positivi, ma y_hat potrebbe avere piccoli residui negativi)
    if np.any(y_fitted <= 0):
        print("Warning: Alcuni valori predetti sono <= 0. Applico il valore assoluto per i pesi.")
        y_fitted = np.abs(y_fitted) + 1e-5 # Small offset per stabilità

    # 3. Calcolo dei pesi: w_i = 1 / sigma_i^2
    # Assumendo che la deviazione standard cresca linearmente con y_hat:
    weights = 1.0 / (y_fitted ** 2)

    # 4. Fit del modello WLS
    # Recuperiamo i dati originali direttamente dall'oggetto ols_model
    y_train = ols_model.model.endog
    X_train = ols_model.model.exog
    
    wls_model = sm.WLS(y_train, X_train, weights=weights).fit()
    
    print("Summary WLS:")
    print(wls_model.summary())
    
    return wls_model

def plot_residui(model, model_name="Model"):
    residuals = model.resid
    fitted = model.fittedvalues
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=fitted, y=residuals, ax=ax, color='steelblue', edgecolor='black')
    ax.axhline(0, color='red', linestyle='--', lw=1)
    ax.set_title(f"Residuals vs Fitted Values: {model_name}", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Fitted Values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./img/{model_name.lower().replace(' ', '_')}_residuals.png", dpi=300, bbox_inches='tight')
    plt.show()

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


def breusch_pagan_test(model, model_name="Model"):
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
        model.resid,
        model.model.exog
    )
    
    print(f"\n=== Test di Breusch-Pagan: {model_name} ===")
    print(f"LM statistic: {lm_stat:.4f}")
    print(f"LM p-value:   {lm_pvalue:.4f}")
    print(f"F statistic:  {f_stat:.4f}")
    print(f"F p-value:    {f_pvalue:.4f}")
    
    if lm_pvalue < 0.05:
        print("Conclusione: evidenza di eteroschedasticità.")
    else:
        print("Conclusione: non emerge eteroschedasticità.")

def white_test(model, model_name="Model"):
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(
        model.resid,
        model.model.exog
    )
    
    print(f"\n=== Test di White: {model_name} ===")
    print(f"LM statistic: {lm_stat:.4f}")
    print(f"LM p-value:   {lm_pvalue:.4f}")
    print(f"F statistic:  {f_stat:.4f}")
    print(f"F p-value:    {f_pvalue:.4f}")
    
    if lm_pvalue < 0.05:
        print("Conclusione: evidenza di eteroschedasticità.")
    else:
        print("Conclusione: non emerge eteroschedasticità.")

import statsmodels.api as sm

def fit_robust_models(ols_model, model_name="Target"):
    """
    Esegue sia WLS (pesi basati su y_hat) che FGLS (pesi stimati dai residui)
    partendo da un modello OLS.
    """
    # 0. Recupero dati originali
    X = ols_model.model.exog
    y = ols_model.model.endog
    names = ols_model.model.exog_names # Manteniamo i nomi delle variabili
    
    print(f"\n" + "="*60)
    print(f" ANALISI ROBUSTA PER: {model_name}")
    print("="*60)

    # --- 1. Calcolo WLS (Pesi Proporzionali) ---
    # Assunzione: la varianza cresce con il valore predetto
    y_hat = np.abs(ols_model.fittedvalues) + 1e-5
    weights_wls = 1.0 / (y_hat**2)
    model_wls = sm.WLS(y, X, weights=weights_wls).fit()
    
    # --- 2. Calcolo FGLS (Feasible GLS) ---
    # Assunzione: la varianza ha una struttura complessa legata alle X
    resid_2 = ols_model.resid**2
    # Regrediamo il log dei residui^2 sulle X per trovare la struttura dell'errore
    log_resid2_model = sm.OLS(np.log(resid_2), X).fit()
    weights_fgls = 1.0 / np.exp(log_resid2_model.fittedvalues)
    model_fgls = sm.WLS(y, X, weights=weights_fgls).fit()

    # --- 3. Output di Confronto ---
    print(f"{'Metodo':<15} | {'R-squared':<10} | {'F-statistic':<12}")
    print("-" * 45)
    print(f"{'OLS':<15} | {ols_model.rsquared:.4f}     | {ols_model.fvalue:.2f}")
    print(f"{'WLS':<15} | {model_wls.rsquared:.4f}     | {model_wls.fvalue:.2f}")
    print(f"{'FGLS':<15} | {model_fgls.rsquared:.4f}     | {model_fgls.fvalue:.2f}")

    return model_wls, model_fgls

def confronta_errori_standard(ols_std, ols_white, model_name="Model"):
    """Confronta SE classici vs HC3 e calcola il Delta %."""
    comparison = pd.DataFrame({
        'Coefficiente': ols_std.params.round(4),
        'SE_Classico':  ols_std.bse.round(5),
        'SE_Robusto_HC3': ols_white.bse.round(5),
        'Delta_SE_%':   ((ols_white.bse - ols_std.bse) / ols_std.bse * 100).round(1)
    })
    print(f"\n>>> ANALISI ROBUSTEZZA SE: {model_name} <<<")
    print(comparison)
    return comparison

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
    #plot_residui(model_heating, "Heating Load")
    #plot_residui(model_cooling, "Cooling Load")

    import numpy as np
    y_heating_log = np.log(data["Heating Load"]) 
    y_cooling_log = np.log(data["Cooling Load"])

    X_base = X.copy() 
    X_heat = X_base.drop(columns=["WallxHeight"])
    model_heating_log = sm.OLS(y_heating_log, X_heat).fit()
    model_heating_log_white = sm.OLS(y_heating_log, X_heat).fit(cov_type='HC3')
    model_cooling_log = sm.OLS(y_cooling_log, X).fit()
    model_cooling_log_white = sm.OLS(y_cooling_log, X).fit(cov_type='HC3')
    model_heating_wls, model_heating_fgls = fit_robust_models(model_heating_log, "Heating Load Log")
    model_cooling_wls, model_cooling_fgls = fit_robust_models(model_cooling_log, "Cooling Load Log")
    # Stampiamo i summary
    #print("=== SUMMARY: HEATING LOAD (LOG) ===")
    #print(model_heating_log.summary())
    #print("=== SUMMARY: HEATING LOAD (LOG) WLS ===")
    #print(model_heating_wls.summary())
    #print("\n=== SUMMARY: COOLING LOAD (LOG) ===")
    #print(model_cooling_log.summary())
    #print("=== SUMMARY: COOLING LOAD (LOG) WLS ===")  
    #print(model_cooling_wls.summary())
    #print("\n=== SUMMARY: HEATING LOAD (LOG) FGLS ===")
    #print(model_heating_fgls.summary())
    #print("=== SUMMARY: COOLING LOAD (LOG) FGLS ===")
    #print(model_cooling_fgls.summary())

    print("=== SUMMARY: HEATING LOAD (LOG) with White's Robust SE ===")
    print(model_heating_log_white.summary())
    print("\n=== SUMMARY: COOLING LOAD (LOG) with White's Robust SE ===")
    print(model_cooling_log_white.summary())

    #plot_residui(model_heating_wls, "Heating Load WLS")
    #plot_residui(model_cooling_wls, "Cooling Load WLS")
    #plot_residui(model_heating_fgls, "Heating Load FGLS")
    #plot_residui(model_cooling_fgls, "Cooling Load FGLS")
    
    wald_test = model_heating_fgls.wald_test(np.eye(len(model_heating_fgls.params))[1:])
    print(f"Statistica Wald (Chi2): {wald_test.statistic.item():.2f}")
    print(f"p-value Wald: {wald_test.pvalue:.4f}")
    # 3. Confronto SE
    confronto_h = confronta_errori_standard(model_heating_log, model_heating_log_white, "Heating Load")
    print("\n" + "="*60)
    print("8======================DConfronto DIO PORCO: ",confronto_h)

    # --- COOLING LOAD ---
    # 1. Modelli OLS (Base e HC3)
    model_cooling_log = sm.OLS(y_cooling_log, X).fit()
    model_cooling_log_white = sm.OLS(y_cooling_log, X).fit(cov_type='HC3')

    # 2. Modelli GLS (WLS e FGLS)
    model_cooling_wls, model_cooling_fgls = fit_robust_models(model_cooling_log, "Cooling")

    # 3. Confronto SE
    confronto_c = confronta_errori_standard(model_cooling_log, model_cooling_log_white, "Cooling Load")
    print("\n" + "="*60)
    print("Confronto: ",confronto_c)

    breusch_pagan_test(model_heating_log_white, "Heating Load WLS")
    breusch_pagan_test(model_cooling_log_white, "Cooling Load WLS")
    white_test(model_heating_log_white, "Heating Load WLS") 
    white_test(model_cooling_log_white, "Cooling Load WLS")
    #white_test(model_heating_log, "Heating Load Log OLS")
    #white_test(model_cooling_log, "Cooling Load Log OLS")
    #white_test(model_heating_wls, "Heating Load WLS")
    #white_test(model_cooling_wls, "Cooling Load WLS")
    #white_test(model_heating_fgls, "Heating Load FGLS")
    #white_test(model_cooling_fgls, "Cooling Load FGLS")
# Energy-Efficient-Linear-Regressor

## How to Start

### 1) Copiare (clonare) la repository
```bash
git clone https://github.com/<tuo-username>/Energy-Efficient-Linear-Regressor.git .
```

### 2) Creare e attivare un ambiente virtuale (Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Installare le dipendenze
```bash
pip install -r requirements.txt
```

### 4) Configurare le variabili ambiente
Crea un file `.env` nella root del progetto con:
```env
DATASET_PATH=/percorso/assoluto/al/dataset.csv
```

### 5) Avviare il progetto
```bash
python3 main.py
```

Le immagini generate verranno salvate nella cartella `img/` (o nella cartella impostata nel codice).

## Note utili
- Assicurati che il percorso in `DATASET_PATH` sia corretto.
- Se `img/` non esiste, creala prima dell'esecuzione:
```bash
mkdir -p img
```
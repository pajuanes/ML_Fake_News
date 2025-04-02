import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_sample(
    input_path,
    output_path,
    stratify_col=None,
    frac=None,
    n=None,
    random_state=42
):
    """
    Crea un sample del dataset original y lo guarda como nuevo CSV.

    Args:
        input_path (str): Ruta del archivo CSV original.
        output_path (str): Ruta donde se guardará el nuevo sample CSV.
        stratify_col (str): Columna para estratificar (por ejemplo: 'label').
        frac (float): Fracción del dataset original a tomar (por ejemplo: 0.1 para 10%).
        n (int): Número exacto de filas a muestrear (incompatible con frac).
        random_state (int): Semilla aleatoria para reproducibilidad.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"El archivo {input_path} no existe.")

    df = pd.read_csv(input_path)

    if frac:
        if stratify_col:
            df_sample, _ = train_test_split(
                df,
                train_size=frac,
                stratify=df[stratify_col],
                random_state=random_state
            )
        else:
            df_sample = df.sample(frac=frac, random_state=random_state)

    elif n:
        if stratify_col:
            df_sample, _ = train_test_split(
                df,
                train_size=n,
                stratify=df[stratify_col],
                random_state=random_state
            )
        else:
            df_sample = df.sample(n=n, random_state=random_state)
    else:
        raise ValueError("Debes indicar 'frac' o 'n'.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sample.to_csv(output_path, index=False)
    print(f"Sample guardado en: {output_path} ({len(df_sample)} registros)")

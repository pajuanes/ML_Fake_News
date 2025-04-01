import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm

class PlotUtils:
    # Definimos un constructor
    def __init__(self, output_dir="../img"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    # Método que generará gráficos por distribuciones
    def plot_text_length_distributions(self, df, text_col, prefix="text", plot_words=True, plot_chars=True, show=False):
        word_count_col = f"{prefix}_word_count"
        char_count_col = f"{prefix}_char_count"

        df[word_count_col] = df[text_col].apply(lambda x: len(str(x).split()))
        df[char_count_col] = df[text_col].apply(lambda x: len(str(x)))

        if plot_words:
            fig = plt.figure()
            sns.histplot(df[word_count_col], bins=100, kde=True)
            plt.title(f"Longitud de {prefix} (palabras)")
            plt.xlabel("Nº de palabras")
            plt.ylabel("Frecuencia")
            if show:
                plt.show()
            fig.savefig(os.path.join(self.output_dir, f"{prefix}_word_count.png"))
            plt.close(fig)

        if plot_chars:
            fig = plt.figure()
            sns.histplot(df[char_count_col], bins=100, kde=True)
            plt.title(f"Longitud de {prefix} (caracteres)")
            plt.xlabel("Nº de caracteres")
            plt.ylabel("Frecuencia")
            if show:
                plt.show()
            fig.savefig(os.path.join(self.output_dir, f"{prefix}_char_count.png"))
            plt.close(fig)

    # Método que generará un gráfico Boxplot por clases
    def plot_boxplot_by_class(self, df, x_col, y_col, title=None, filename=None,
                          xlabel=None, ylabel=None, plot_type="box", show=False):
        """
        Genera un gráfico comparando una métrica numérica entre clases (boxplot o violinplot).

        Args:
            df (DataFrame): Dataset
            x_col (str): Columna categórica (eje X)
            y_col (str): Columna numérica (eje Y)
            title (str): Título del gráfico
            filename (str): Nombre del archivo a guardar (ej: 'plot.png')
            xlabel (str): Etiqueta del eje X (opcional)
            ylabel (str): Etiqueta del eje Y (opcional)
            plot_type (str): 'box' o 'violin'
            show (bool): Mostrar el gráfico por pantalla
        """
        fig = plt.figure()

        # Gráfico según tipo
        if plot_type == "box":
            sns.boxplot(data=df, x=x_col, y=y_col)
        elif plot_type == "violin":
            sns.violinplot(data=df, x=x_col, y=y_col)
        else:
            raise ValueError("El argumento 'plot_type' debe ser 'box' o 'violin'.")

        plt.xlabel(xlabel if xlabel else x_col.capitalize())
        plt.ylabel(ylabel if ylabel else y_col.replace("_", " ").capitalize())

        if title:
            plt.title(title)
        else:
            plt.title(f"{y_col} por {x_col}")

        if filename:
            path = os.path.join(self.output_dir, filename)
            fig.savefig(path)

        if show:
            plt.show()

        plt.close(fig)


class Texts:
    def clean_text_basic(text, to_lower=True):
        """
        Limpieza básica de texto:
        - Elimina saltos de línea, tabs y símbolos especiales
        - Normaliza espacios en blanco
        - Opcionalmente convierte a minúsculas

        Args:
            text (str): Texto original
            to_lower (bool): Si True, convierte a minúsculas

        Returns:
            str: Texto limpio
        """
        if not isinstance(text, str):
            text = str(text)

        text = re.sub(r"[\n\r\t]", " ", text)         # Saltos y tabuladores
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)   # Elimina símbolos especiales
        text = re.sub(r"\s+", " ", text)              # Normaliza espacios
        text = text.strip()

        if to_lower:
            text = text.lower()

        return text
    
    import re

# Descargar stopwords si no están
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# Inicializar modelo spaCy (idioma inglés por defecto)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class Texts:
    @staticmethod
    def clean_text_basic(text, to_lower=True):
        if not isinstance(text, str):
            text = str(text)

        text = re.sub(r"[\n\r\t]", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        if to_lower:
            text = text.lower()

        return text

    @staticmethod
    def tokenize(text):
        """
        Divide el texto en palabras (tokens) usando .split().
        """
        return text.split()

    @staticmethod
    def remove_stopwords(tokens, language="english"):
        """
        Elimina stopwords de una lista de tokens.

        Args:
            tokens (List[str]): Lista de palabras.
            language (str): Idioma ('english', 'spanish', etc.)

        Returns:
            List[str]: Lista sin stopwords.
        """
        stop_words = set(stopwords.words(language))
        return [word for word in tokens if word not in stop_words]

    @staticmethod
    def lemmatize_bulk_blocked(texts, to_lower=True, remove_stop=True,
                                batch_size=50, block_size=5000,
                                output_dir="../models/text_lemma_blocks",
                                final_output_path="../models/text_lemmatized.pkl",
                                desc="Lematizando"):
        """
        Lematiza textos en bloques, guarda cada bloque por separado y permite continuar si se detiene.

        Args:
            texts (List[str]): Lista de textos
            to_lower (bool): Convertir a minúsculas
            remove_stop (bool): Eliminar stopwords
            batch_size (int): Tamaño de lote spaCy
            block_size (int): Nº de textos por bloque
            output_dir (str): Carpeta donde se guardan los bloques .pkl
            final_output_path (str): Archivo final que agrupa todos los bloques
            desc (str): Descripción para la barra de progreso
        """
        os.makedirs(output_dir, exist_ok=True)

        total = len(texts)
        processed_blocks = [int(f.split("_")[-1].replace(".pkl", "")) 
                            for f in os.listdir(output_dir) if f.startswith("text_lemma_block_")]

        start_block = max(processed_blocks) + 1 if processed_blocks else 0
        total_blocks = (total + block_size - 1) // block_size

        print(f"Retomando desde el bloque {start_block} de {total_blocks}")

        for i in tqdm(range(start_block, total_blocks), desc=f"{desc} (por bloques)"):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, total)
            block = texts[start_idx:end_idx]
            block_lemmas = []

            for doc in nlp.pipe(block, batch_size=batch_size):
                lemmas = [
                    token.lemma_ for token in doc
                    if not token.is_punct and not token.is_space and (not token.is_stop if remove_stop else True)
                ]
                text = " ".join(lemmas)
                block_lemmas.append(text.lower() if to_lower else text)

            # Guardar bloque
            df_block = pd.DataFrame({ "text_lemma": block_lemmas })
            path_block = os.path.join(output_dir, f"text_lemma_block_{i}.pkl")
            df_block.to_pickle(path_block)
            print(f"Bloque {i} guardado: {path_block}")

        # Al final: unir todos los bloques y guardar resultado
        print("\nUniendo bloques...")
        all_blocks = []
        for i in range(total_blocks):
            path = os.path.join(output_dir, f"text_lemma_block_{i}.pkl")
            df_block = pd.read_pickle(path)
            all_blocks.append(df_block)

        df_final = pd.concat(all_blocks, ignore_index=True)
        df_final.to_pickle(final_output_path)
        print(f"Archivo final guardado en: {final_output_path}")

        return df_final["text_lemma"].tolist()

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

class ModelEvaluator:
    def __init__(self):
        self.model_metrics = []

    def add_metrics(self, model_name, y_true, y_pred):
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-score": f1_score(y_true, y_pred)
        }
        self.model_metrics.append(metrics)

    def get_metrics_df(self):
        return pd.DataFrame(self.model_metrics)

    def save_metrics(self, output_dir="../models", filename="model_metrics_comparison.csv"):
        os.makedirs(output_dir, exist_ok=True)
        df = self.get_metrics_df()
        path_csv = os.path.join(output_dir, filename)
        df.to_csv(path_csv, index=False)
        print(f"Métricas guardadas en CSV: {path_csv}")

        path_json = path_csv.replace(".csv", ".json")
        df.to_json(path_json, orient="records", indent=2)
        print(f"Métricas guardadas en JSON: {path_json}")
    
    # Método para mostrar gráficamente la matriz de confusión de un modelo concreto
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Modelo"):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - {model_name}")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.show()

    # Método para mostrar gráficamente todas las matrices de confusión
    def plot_all_confusion_matrices(self, y_true_list, y_pred_list, model_names, labels=["Real", "Fake"], save=False,
                                 output_dir="../img", filename="confusion_matrices.png"):
        """
        Muestra todas las matrices de confusión de los modelos en una única figura (subplots).

        Args:
            y_true_list (List[np.array]): Lista de vectores y_true para cada modelo
            y_pred_list (List[np.array]): Lista de vectores y_pred para cada modelo
            model_names (List[str]): Lista de nombres de los modelos
            labels (List[str]): Etiquetas de clase (por defecto: Real, Fake)
            save (bool): Si True, guarda la imagen
            output_dir (str): Carpeta de salida
            filename (str): Nombre del archivo
        """
        import math
        num_models = len(model_names)
        cols = 2
        rows = math.ceil(num_models / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for idx, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], xticklabels=labels, yticklabels=labels)
            axes[idx].set_title(f"Matriz de Confusión - {name}")
            axes[idx].set_xlabel("Predicción")
            axes[idx].set_ylabel("Real")

        # Quitar subgráficos vacíos si hay número impar
        for i in range(idx + 1, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            plt.savefig(path)
            print(f"Confusion matrices guardadas en: {path}")

        plt.show()
    
    # Método para mostrar un Classification Report
    def print_classification_report(self, y_true, y_pred, target_names=["Real", "Fake"]):
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Método para mostrar graficamente el Classification Report de un modelo concreto
    def plot_classification_report(self, y_true, y_pred, model_name="Modelo", target_names=["Real", "Fake"],
                                save=False, output_dir="../img", filename=None):
        """
        Visualiza gráficamente el classification report (precision, recall, F1-score) de un solo modelo.

        Args:
            y_true (array): Valores reales
            y_pred (array): Valores predichos
            model_name (str): Nombre del modelo
            target_names (list): Clases
            save (bool): Si True, guarda la imagen
            output_dir (str): Carpeta donde guardar el gráfico
            filename (str): Nombre del archivo. Si no se pasa, se genera automáticamente.
        """
        report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose().loc[target_names]

        ax = df_report[["precision", "recall", "f1-score"]].plot(kind="bar", ylim=(0, 1), figsize=(8, 4), colormap="Set2")
        plt.title(f"Classification Report - {model_name}")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.grid(axis="y")
        plt.tight_layout()

        if save:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                filename = f"classification_report_{model_name.replace(' ', '_').lower()}.png"
            path = os.path.join(output_dir, filename)
            plt.savefig(path)
            print(f"Reporte visual guardado en: {path}")

        plt.show()
    
    # Método para mostrar graficamente los Classification Report de todos los modelos
    def plot_classification_reports(self, y_true_list, y_pred_list, model_names, target_names=["Real", "Fake"],
                                 save=False, output_dir="../img", filename="classification_reports.png"):
        """
        Visualiza classification reports de múltiples modelos en subplots (2 por fila), con etiquetas numéricas.

        Args:
            y_true_list (List): Lista de vectores y_true
            y_pred_list (List): Lista de vectores y_pred
            model_names (List[str]): Lista de nombres de los modelos
            target_names (List[str]): Clases
            save (bool): Si True, guarda la imagen
            output_dir (str): Carpeta de salida
            filename (str): Nombre del archivo PNG
        """
        import math

        num_models = len(model_names)
        cols = 2
        rows = math.ceil(num_models / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for idx, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
            report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
            df_report = pd.DataFrame(report_dict).transpose().loc[target_names]

            ax = axes[idx]
            plot = df_report[["precision", "recall", "f1-score"]].plot(kind="bar", ylim=(0, 1), ax=ax, legend=True)
            ax.set_title(name)
            ax.set_ylabel("Score")
            ax.grid(axis="y")
            ax.set_xticklabels(target_names, rotation=0)

            # Añadir etiquetas numéricas encima de cada barra
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", padding=3)

        # Eliminar ejes vacíos si hay número impar
        for i in range(idx + 1, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            plt.savefig(path)
            print(f"Classification reports guardados en: {path}")

        plt.show()
    
    def plot_metrics_comparison(self, save=False, output_dir="../img", filename="model_metrics_barplot.png"):
        """
        Genera una gráfica de barras para comparar Accuracy, Precision, Recall y F1-score por modelo.

        Args:
            save (bool): Si True, guarda la imagen como PNG.
            output_dir (str): Carpeta de salida.
            filename (str): Nombre del fichero de imagen.
        """
        df = self.get_metrics_df()
        df_plot = df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]]

        ax = df_plot.plot(kind="bar", figsize=(10, 6), colormap="Set2")
        plt.title("Comparación de métricas por modelo")
        plt.ylabel("Valor")
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.legend(loc="lower right")
        plt.xticks(rotation=0)
        plt.tight_layout()

        if save:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            plt.savefig(path)
            print(f"Gráfico guardado en: {path}")

        plt.show()

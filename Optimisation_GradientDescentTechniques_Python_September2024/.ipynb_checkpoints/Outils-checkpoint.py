import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from IPython.display import display









### Fonctions pour la régression logistique

# La sigmoîde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fonction pour initialiser un vecteur de poids de départ ( au hasard, empiriquement efficace)
def random_initialize_parameters(n_features):
    return np.random.randn(n_features + 1)

# Fonction pour prédire la probabilité de l'appartenance à la classe 1 des observations
def predict(X, w):
    return sigmoid(np.dot(X, w[1:]) + w[0])

# Fonction pour calculer la perte logistique, definie en tant que binary cross entropy loss dans ce cas
def cost_function(X, y, w, epsilon=1e-8): # notons le rajout d'un petit epsilon pour éviter les divisions par zéro
    y_pred = predict(X, w)
    cost = - (1 / len(y)) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    return cost



######################################################################################################################



### Fonctions de calcul de métrique de bonne prédiction

# Calcul de l'accuracy
def calculate_accuracy(y_test, y_pred):
    """
    Calcule et retourne l'accuracy entre y_test et y_pred.

    Args:
    y_test : les vraies étiquettes
    y_pred : les étiquettes prédites

    Returns:
    L'accuracy
    """
    accuracy = np.mean(y_pred == y_test)
    return accuracy

# Calcul de la précision
def calculate_precision(y_test, y_pred):
    """
    Calcule et retourne la précision (precision) entre y_test et y_pred.

    Args:
    y_test : les vraies étiquettes
    y_pred : les étiquettes prédites

    Returns:
    La précision (precision)
    """
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    predicted_positives = np.sum(y_pred == 1)
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    return precision

# Calcul du recall
def calculate_recall(y_test, y_pred):
    """
    Calcule et retourne le rappel (recall) entre y_test et y_pred.

    Args:
    y_test : les vraies étiquettes
    y_pred : les étiquettes prédites

    Returns:
    Le rappel (recall)
    """
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    actual_positives = np.sum(y_test == 1)
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    return recall

# Calcul du F1-score
def calculate_f1_score(y_test, y_pred):
    """
    Calcule et retourne le F1-score entre y_test et y_pred.

    Args:
    y_test : les vraies étiquettes
    y_pred : les étiquettes prédites

    Returns:
    Le F1-score
    """
    precision = calculate_precision(y_test, y_pred)
    recall = calculate_recall(y_test, y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

# Tracer la courbe AUC-ROC
def plot_roc_curve(y_test, y_prob):
    """
    Trace la courbe ROC et calcule l'AUC.

    Args:
    y_test : les vraies étiquettes
    y_prob : les probabilités prédites (output de sigmoid par exemple)

    Returns:
    L'AUC de la courbe ROC
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    return roc_auc

# Matrice de Confusion
def calculate_confusion_matrix(y_test, y_pred):
    """
    Calcule et retourne la matrice de confusion entre y_test et y_pred sous forme de DataFrame avec des intitulés clairs.

    Args:
    y_test : les vraies étiquettes
    y_pred : les étiquettes prédites

    Returns:
    La matrice de confusion sous forme de DataFrame
    """
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))
    
    # Créer la matrice sous forme de DataFrame avec des intitulés clairs
    confusion_matrix_df = pd.DataFrame(
        [[true_negatives, false_positives], [false_negatives, true_positives]],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    
    return confusion_matrix_df

# Calculer toutes les métriques avec nos outils
def calculate_all_metrics_for_one(y_test, y_pred): # pour un algo
    """
    Calcule toutes les métriques de prédiction (accuracy, precision, recall, f1-score, confusion matrix)
    et les retourne sous forme de dictionnaire.

    Args:
    y_test : Les vraies étiquettes.
    y_pred : Les étiquettes prédites (après application du seuil de 0.5).

    Returns:
    results_metrics : dictionnaire avec accuracy, precision, recall, f1_score, confusion_matrix
    """
    
    # Initialisation des dictionnaires pour stocker les résultats
    results_metrics = {
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1_score': None,
        'confusion_matrix': None
    }
    
    # Calcul de l'accuracy
    results_metrics['accuracy'] = calculate_accuracy(y_test, y_pred)
    
    # Calcul de la précision
    results_metrics['precision'] = calculate_precision(y_test, y_pred)
    
    # Calcul du rappel (recall)
    results_metrics['recall'] = calculate_recall(y_test, y_pred)
    
    # Calcul du F1-score
    results_metrics['f1_score'] = calculate_f1_score(y_test, y_pred)
    
    # Calcul de la matrice de confusion sous forme de DataFrame
    results_metrics['confusion_matrix'] = calculate_confusion_matrix(y_test, y_pred)
    
    return results_metrics

# Calculer toutes les métriques pour un ensemble : 
def calculate_all_metrics_for_all(y_test, y_pred_dict): # pour un dictionnaire de y_pred
    """
    Calcule toutes les métriques de prédiction (accuracy, precision, recall, f1-score, confusion matrix)
    pour chaque algorithme dans y_pred_dict et les retourne sous forme de dictionnaire de dictionnaires.

    Args:
    y_test : Les vraies étiquettes.
    y_pred_dict : Un dictionnaire où les clés sont les noms des algorithmes et les valeurs sont les étiquettes prédites.

    Returns:
    all_results_metrics : dictionnaire de dictionnaires avec accuracy, precision, recall, f1_score, confusion_matrix pour chaque  algorithme.
    """
    
    # Initialisation du dictionnaire pour stocker les résultats pour chaque algorithme
    all_results_metrics = {}

    # Pour chaque algorithme dans y_pred_dict
    for algo, y_pred in y_pred_dict.items():
        # Initialisation des métriques pour l'algorithme courant
        results_metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'confusion_matrix': None
        }

        # Calcul des métriques
        results_metrics['accuracy'] = calculate_accuracy(y_test, y_pred)
        results_metrics['precision'] = calculate_precision(y_test, y_pred)
        results_metrics['recall'] = calculate_recall(y_test, y_pred)
        results_metrics['f1_score'] = calculate_f1_score(y_test, y_pred)
        results_metrics['confusion_matrix'] = calculate_confusion_matrix(y_test, y_pred)

        # Stocker les résultats pour cet algorithme
        all_results_metrics[algo] = results_metrics

    return all_results_metrics



########################################################################################################################



### Fonctions Graphiques

# Fonction pour plotter les pertes à partir d'un dictionnaire
def plot_losses(result_dict, index_min=0, index_max=1000, plot=[], dontplot=[]):
    """
    Prend un dictionnaire de résultats et trace les courbes de pertes pour chaque algorithme.

    Args:
    result_dict : dictionnaire où les clés sont les noms des algorithmes et les valeurs sont des listes de pertes.
    index_min : index de début pour les points à afficher.
    index_max : index de fin pour les points à afficher.
    plot : liste des algorithmes à inclure (si None, tous sauf ceux dans dontplot).
    dontplot : liste des algorithmes à exclure.
    """
    plt.figure(figsize=(10, 6))
    
    # Si une liste `plot` est spécifiée, on utilise uniquement ces clés
    if len(plot) != 0:
        keys_to_plot = [key for key in result_dict if key in plot]
    else:
        # Si `plot` n'est pas spécifiée, on exclut les algorithmes dans `dontplot`
        keys_to_plot = [key for key in result_dict if key not in dontplot]

    # Tracer chaque algorithme sélectionné
    for key in keys_to_plot:
        costs = result_dict[key][index_min:index_max]  # Limiter les valeurs à afficher
        iterations = list(range(index_min, index_min + len(costs)))  # Générer les indices correspondants
        plt.plot(iterations, costs, label=key)

    plt.xlabel('Itérations')
    plt.ylabel('Coût')
    plt.title('Comportement de la perte')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotter un dataframe de métriques de prédiction pour plusieurs algorithmes :
import pandas as pd

def display_metrics_dataframe(all_algos_metrics):
    """
    Convertit le dictionnaire des résultats des métriques en DataFrame et l'affiche.

    Args:
    all_algos_metrics : dictionnaire contenant les métriques pour chaque algorithme.
    """
    # Initialiser une liste vide pour stocker les données sous forme de lignes
    metrics_data = []

    # Extraire les noms des algorithmes et les métriques correspondantes
    for algo, metrics in all_algos_metrics.items():
        # Créer une liste avec les valeurs des métriques pour chaque algorithme
        row = [
            algo,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        # Ajouter la ligne à la liste des données
        metrics_data.append(row)

    # Créer un DataFrame à partir des données et des noms de colonnes
    metrics_df = pd.DataFrame(
        metrics_data, 
        columns=['Algorithme', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    )
    
    # Afficher le DataFrame
    display(metrics_df)

# Plotter les matrices de confusion 
def plot_confusion_matrices(all_algos_metrics):
    """
    Trace les matrices de confusion pour chaque algorithme, 4 matrices par ligne.

    Args:
    all_algos_metrics : dictionnaire contenant les métriques pour chaque algorithme, y compris la matrice de confusion.
    """
    # Nombre total d'algorithmes
    num_algos = len(all_algos_metrics)
    
    # Calcul du nombre de lignes nécessaires pour avoir 4 matrices par ligne
    num_rows = (num_algos + 3) // 4  # +3 pour arrondir vers le haut
    
    # Initialisation de la figure avec des sous-graphiques (4 colonnes)
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))  # Ajuster la taille en fonction des lignes
    
    # Aplatir l'array des axes pour pouvoir les itérer facilement
    axes = axes.flatten()
    
    # Itérer sur les algorithmes et les matrices de confusion
    for i, (algo, metrics) in enumerate(all_algos_metrics.items()):
        # Extraire la matrice de confusion
        cm = metrics['confusion_matrix']
        
        # Tracer la heatmap de la matrice de confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        
        # Mettre le nom de l'algorithme en titre du graphique
        axes[i].set_title(f"Confusion Matrix: {algo}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Masquer les sous-graphiques inutilisés s'il y en a moins de 4 dans la dernière ligne
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Ajuster l'espacement des sous-graphiques pour éviter la superposition
    plt.tight_layout()
    plt.show()



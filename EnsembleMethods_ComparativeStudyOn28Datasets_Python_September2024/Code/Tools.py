
# Librairies
import numpy as np
import pandas as pd
import csv
import time
import pickle
from random import random, seed
import seaborn as sns
from holoviews.ipython import display
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from memory_profiler import memory_usage
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE






# Fonction pour sauvegarder un dictionnaire dans un fichier pickle
def save_dict_to_pickle(dict_obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict_obj, file)

# Fonction pour charger un dictionnaire à partir d'un fichier pickle
def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def datasets_to_dataframes(file_list):
    """
    Convertit une liste de fichiers datasets en une liste de DataFrames pandas.

    Args:
        file_list (list): Liste des noms de fichiers datasets.

    Returns:
        list: Liste de DataFrames pandas, chacun contenant les données d'un dataset.
    """
    dataframes = []

    for dataset in file_list:
        # Charger le dataset avec la fonction data_recovery (supposée définie ailleurs)
        X, y = data_recovery(dataset)

        # Convertir X et y en DataFrame
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y, columns=['target'])

        # Combiner X et y dans un seul DataFrame
        df = pd.concat([df_X, df_y], axis=1)
        dataframes.append(df)

    return dataframes


def analyze_datasets(dataframes, files_list):
    """
    Analyse une liste de DataFrames pour obtenir des informations récapitulatives.

    Args:
        dataframes (list): Liste de DataFrames à analyser.

    Returns:
        pd.DataFrame: DataFrame contenant les informations récapitulatives pour chaque dataset.
    """
    summary = []

    def detect_outliers(df):
        """
        Détecte les valeurs aberrantes dans un DataFrame en utilisant la méthode IQR.

        Args:
            df (pd.DataFrame): DataFrame à analyser.

        Returns:
            list: Indices des lignes contenant des valeurs aberrantes.
        """
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        return df[outliers].index.tolist()

    for df, dataset in zip(dataframes, files_list):
        # Analyser les types de colonnes
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Vérifier les valeurs manquantes et dupliquées
        missing_indices = df[df.isnull().any(axis=1)].index.tolist()
        duplicated_indices = df[df.duplicated()].index.tolist()

        # Vérifier la variance des colonnes
        variances = df.var()
        low_variance_cols = variances[variances < 1e-10].index.tolist()

        # Vérifier la colinéarité
        correlation_matrix = df.corr().abs()
        high_corr_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns
                           if i != j and correlation_matrix.loc[i, j] > 0.999]

        # Détecter les valeurs aberrantes
        outlier_indices = detect_outliers(df[num_cols])

        # Analyser la distribution des classes cibles
        class_ratios = df['target'].value_counts(normalize=True).to_dict()

        # Créer un dictionnaire avec les informations récapitulatives
        info = {
            'dataset_name': dataset,
            'dataset_shape': df.shape,
            'num_numeric_cols': len(num_cols),
            'num_cat_cols': len(cat_cols),
            'missing_indices': missing_indices,
            'duplicated_indices': duplicated_indices,
            'low_variance_cols': low_variance_cols,
            'high_corr_pairs': high_corr_pairs,
            'outlier_indices': outlier_indices,
            'class_ratios': class_ratios if len(class_ratios) > 1 else "continuous_target"
        }

        summary.append(info)

    # Convertir la liste d'infos en DataFrame
    summary_df = pd.DataFrame(summary)

    return summary_df


def process_dataframes(dataframes, summary_df, drop_na=True, drop_duplicates=True,
                       drop_low_variance=True, drop_redundant=True, drop_outliers=False):
    """
    Traite une liste de DataFrames en fonction des informations récapitulatives et des options spécifiées.

    Args:
        dataframes (list): Liste de DataFrames à traiter.
        summary_df (pd.DataFrame): DataFrame contenant les informations récapitulatives.
        drop_na (bool): Si True, supprime les lignes avec des valeurs manquantes.
        drop_duplicates (bool): Si True, supprime les lignes dupliquées.
        drop_low_variance (bool): Si True, supprime les colonnes à faible variance.
        drop_redundant (bool): Si True, supprime les colonnes redondantes (fortement corrélées).
        drop_outliers (bool): Si True, supprime les valeurs aberrantes.

    Returns:
        list: Liste des DataFrames traités.
    """
    processed_dataframes = []

    for df, summary in zip(dataframes, summary_df.itertuples()):
        # Suppression des valeurs manquantes
        if drop_na and summary.missing_indices:
            df = df.drop(index=summary.missing_indices)

        # Suppression des doublons
        if drop_duplicates and summary.duplicated_indices:
            df = df.drop(index=summary.duplicated_indices)

        # Suppression des colonnes à faible variance
        if drop_low_variance and summary.low_variance_cols:
            df = df.drop(columns=summary.low_variance_cols)

        # Suppression des colonnes redondantes
        if drop_redundant and summary.high_corr_pairs:
            columns_to_keep = set()
            columns_to_drop = set()

            for col1, col2 in summary.high_corr_pairs:
                if col1 not in columns_to_keep and col2 not in columns_to_keep:
                    columns_to_keep.add(col1)
                    columns_to_drop.add(col2)
                elif col1 in columns_to_keep:
                    columns_to_drop.add(col2)
                elif col2 in columns_to_keep:
                    columns_to_drop.add(col1)

            df = df.drop(columns=columns_to_drop)

        # Suppression des valeurs aberrantes
        if drop_outliers and summary.outlier_indices:
            df = df.drop(index=summary.outlier_indices)

        processed_dataframes.append(df)

    return processed_dataframes


def split_dataframes_by_class_ratio(dataframes, summary_df, threshold=0.6):
    """
    Divise la liste des dataframes en deux groupes basés sur le ratio de la classe majoritaire.
    Chaque groupe est ensuite trié par ordre croissant du ratio de la classe "0" ( ici toujours majoritaire).

    Paramètres:
    - dataframes: liste de pandas DataFrames
    - summary_df: pandas DataFrame contenant les informations récapitulatives, incluant 'dataset_name' et 'class_ratios'
    - threshold: float, seuil pour le ratio de la classe majoritaire (par défaut 0.6)

    Retourne:
    - group1: liste de tuples (dataset_name, dataframe) où le ratio de la classe majoritaire <= seuil, trié par ratio de la classe "0"
    - group2: liste de tuples (dataset_name, dataframe) où le ratio de la classe majoritaire > seuil, trié par ratio de la classe "0"
    """
    group1 = []
    group2 = []

    # Création d'un mapping entre le nom du dataset et le dataframe correspondant
    dataset_names = summary_df['dataset_name'].tolist()
    dataset_to_dataframe = {name: df for name, df in zip(dataset_names, dataframes)}

    # Itération sur chaque dataset
    for index, row in summary_df.iterrows():
        dataset_name = row['dataset_name']
        class_ratios = row['class_ratios']

        # Vérifier que ce n'est pas une cible continue
        if class_ratios != "continuous_target":
            # Obtenir le ratio de la classe majoritaire
            majority_class_ratio = max(class_ratios.values())
            class_0_ratio = class_ratios.get(0, 0)  # Obtenir le ratio de la classe "0" (ou 0 si absent)

            # Récupérer le dataframe correspondant
            df = dataset_to_dataframe[dataset_name]

            # Séparer les datasets selon le seuil
            if majority_class_ratio <= threshold:
                group1.append((dataset_name, df, class_0_ratio))  # Ajouter le ratio de la classe 0 au tuple
            else:
                group2.append((dataset_name, df, class_0_ratio))  # Ajouter le ratio de la classe 0 au tuple

    # Trier chaque groupe par ordre croissant du ratio de la classe "0"
    group1 = sorted(group1, key=lambda x: x[2])
    group2 = sorted(group2, key=lambda x: x[2])

    # Retirer le ratio de la classe 0 des résultats pour ne renvoyer que (dataset_name, dataframe)
    group1 = [(name, df) for name, df, _ in group1]
    group2 = [(name, df) for name, df, _ in group2]

    return group1, group2



def create_ratio_table(group1, group2):
    """
    Crée un tableau avec trois colonnes : 'Nom', 'Groupe', 'Ratio classe 1'
    à partir des groupes 1 et 2, trié par ratio classe 1 décroissant.

    Paramètres :
    - group1: Liste de tuples (nom_dataset, dataframe) pour les datasets du groupe 1
    - group2: Liste de tuples (nom_dataset, dataframe) pour les datasets du groupe 2

    Retourne :
    - DataFrame pandas avec les colonnes 'Nom', 'Groupe', 'Ratio classe 1'
    """

    # Liste pour stocker les résultats
    results = []

    # Remplir les résultats pour le groupe 1
    for name, df in group1:
        # Calcul du ratio de la classe "1" dans chaque dataset
        class_1_ratio = df['target'].value_counts(normalize=True).get(1, 0)
        # Ajouter les résultats à la liste
        results.append({'Nom': name, 'Groupe': 'Group 1', 'Ratio classe 1': class_1_ratio})

    # Remplir les résultats pour le groupe 2
    for name, df in group2:
        # Calcul du ratio de la classe "1" dans chaque dataset
        class_1_ratio = df['target'].value_counts(normalize=True).get(1, 0)
        # Ajouter les résultats à la liste
        results.append({'Nom': name, 'Groupe': 'Group 2', 'Ratio classe 1': class_1_ratio})

    # Créer un DataFrame pandas à partir des résultats
    results_df = pd.DataFrame(results)

    # Trier par ratio de classe 1 décroissant
    results_df = results_df.sort_values(by='Ratio classe 1', ascending=False)

    # Retourner le DataFrame
    return results_df


def split_groups_train_test(group1, group2, test_size=0.2, random_state=0):
    """
    Divise chaque DataFrame dans group1 et group2 en ensembles d'entraînement et de test.
    La dernière colonne de chaque DataFrame est utilisée comme colonne cible (labels).

    Retourne un dictionnaire où chaque dataset peut être accédé directement par son nom.
    """
    seed(random_state)
    train_test_data = {"group1": {}, "group2": {}}

    # Split pour group1
    for name, df in group1:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        train_test_data["group1"][name] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    # Split pour group2
    for name, df in group2:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        train_test_data["group2"][name] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    return train_test_data


def basic_fintune_and_evaluate_CV(train_test_data, classifier, param_name, param_values, cv=5, scoring='recall'):
    """
    Évalue chaque dataset dans train_test_data en fine-tunant un paramètre du classificateur à l'aide de GridSearchCV.
    Utilise un pipeline avec RobustScaler pour gérer les outliers et un suivi du temps d'entraînement pour estimer la complexité temporelle.

    Paramètres :
    - train_test_data : Dictionnaire contenant les datasets (X_train, X_test, y_train, y_test) par groupe et nom de dataset.
    - classifier : Le modèle de classification à fine-tuner.
    - param_name : Nom du paramètre à fine-tuner.
    - param_values : Liste des valeurs pour le paramètre à tester.
    - cv : Nombre de folds pour la validation croisée (par défaut 5).
    - scoring : Métrique utilisée pour l'évaluation (par défaut 'accuracy').

    Retourne :
    - results_df : DataFrame contenant les résultats des métriques pour chaque dataset.
    - best_models : Dictionnaire contenant les modèles ajustés pour chaque dataset.
    """

    results = []
    best_models = {}  # Dictionnaire pour stocker les modèles ajustés

    # Parcourir les groupes de datasets
    for group_name in train_test_data:
        # Parcourir chaque dataset dans le groupe
        for dataset_name, data in train_test_data[group_name].items():
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

            # Taille des ensembles de données
            train_size = len(X_train)
            test_size = len(X_test)
            num_positives = sum(y_test == 1)  # Compte des classes positives dans y_test

            # Créer un pipeline avec RobustScaler et le classificateur
            pipeline = Pipeline([
                ('scaler', RobustScaler()),  # Gérer les outliers avec RobustScaler
                ('classifier', classifier)  # Le classificateur à fine-tuner
            ])

            # Configurer GridSearchCV pour la recherche du meilleur paramètre
            param_grid = {f'classifier__{param_name}': param_values}  # Le format pour GridSearch avec pipeline
            # Stratifier les folds, pour sauvegarder les proportions de classe
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            # Application
            grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring=scoring, n_jobs=-1)

            # Suivi du temps d'entraînement
            start_time = time.time()  # Temps avant l'entraînement
            mem_before = memory_usage()[0]  # Utilisation de la mémoire avant l'entraînement

            grid_search.fit(X_train, y_train)  # Entraîner le modèle avec la recherche de grille

            mem_after = memory_usage()[0]  # Utilisation de la mémoire après l'entraînement
            end_time = time.time()  # Temps après l'entraînement

            training_time = end_time - start_time  # Calculer le temps d'entraînement
            memory_used = max(0, mem_after - mem_before)  # S'assurer que la mémoire utilisée n'est pas négative

            # Récupérer le meilleur paramètre et le meilleur modèle
            best_param = grid_search.best_params_[f'classifier__{param_name}']
            best_model = grid_search.best_estimator_

            # Faire des prédictions sur l'ensemble de test
            y_pred = best_model.predict(X_test)

            # Calculer les probabilités pour AUC-ROC et AUC-PR si applicable
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_proba = best_model.decision_function(X_test)
            else:
                y_proba = None

            # Calculer les métriques de performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Calculer AUC-ROC et AUC-PR si y_proba est disponible
            auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            auc_pr = average_precision_score(y_test, y_proba) if y_proba is not None else None

            # Stocker les résultats dans la liste
            results.append({
                'group_name': group_name,
                'dataset_name': dataset_name,
                'train_size': train_size,  # Taille de l'ensemble d'entraînement
                'test_size': test_size,  # Taille de l'ensemble de test
                'test_positives': num_positives,  # Nombre de positifs dans y_test
                'best_param': best_param,  # Meilleur paramètre sélectionné
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'training_time': training_time,  # Temps d'entraînement
                'memory_used': memory_used  # Mémoire utilisée
            })

            # Stocker le modèle ajusté
            best_models[dataset_name] = best_model

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)

    # Calculer les moyennes par groupe pour toutes les métriques
    group_means = results_df.groupby('group_name').mean(numeric_only=True).add_prefix('group_mean_')

    # Fusionner les moyennes avec les résultats
    results_df = results_df.merge(group_means, on='group_name', how='left')

    # Retourner les résultats et les modèles ajustés
    return results_df, best_models

def ensemble_fine_tune_and_evaluate_CV(train_test_data, classifier, param_grid, cv=5, scoring='recall'):
    """
    Évalue chaque dataset dans train_test_data en fine-tunant plusieurs paramètres du classificateur
    en utilisant GridSearchCV. Conçu pour les méthodes ensemblistes (Bagging, Boosting, RandomForest, etc.)
    sans pondération des poids, avec un suivi du temps d'entraînement et de l'utilisation mémoire.

    Paramètres :
    - train_test_data : Dictionnaire contenant les datasets (X_train, X_test, y_train, y_test) par groupe et nom de dataset.
    - classifier : Le modèle d'ensemble à fine-tuner.
    - param_grid : Grille de paramètres à fine-tuner sous forme de dictionnaire (GridSearchCV).
    - cv : Nombre de folds pour la validation croisée (par défaut 5).
    - scoring : Métrique utilisée pour l'évaluation (par défaut 'accuracy').

    Retourne :
    - results_df : DataFrame contenant les résultats des métriques pour chaque dataset.
    - best_models : Dictionnaire contenant les modèles ajustés pour chaque dataset.
    """

    results = []
    best_models = {}  # Dictionnaire pour stocker les modèles ajustés

    # Parcourir les groupes de datasets
    for group_name in train_test_data:
        # Parcourir chaque dataset dans le groupe
        for dataset_name, data in train_test_data[group_name].items():
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

            # Taille des ensembles de données
            train_size = len(X_train)
            test_size = len(X_test)
            num_positives = sum(y_test == 1)  # Compte des classes positives dans y_test

            # Créer un pipeline avec RobustScaler et le classificateur
            pipeline = Pipeline([
                ('scaler', RobustScaler()),  
                ('classifier', classifier)  # Le classificateur d'ensemble à fine-tuner
            ])

            # Stratifier les folds, pour sauvegarder les proportions de classe
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            

            # Configurer GridSearchCV pour la recherche du meilleur paramètre
            grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring=scoring, n_jobs=-1)

            # Suivi du temps d'entraînement
            start_time = time.time()  # Temps avant l'entraînement
            mem_before = memory_usage()[0]  # Utilisation de la mémoire avant l'entraînement

            # Entraîner le modèle avec la recherche de grille
            grid_search.fit(X_train, y_train)

            mem_after = memory_usage()[0]  # Utilisation de la mémoire après l'entraînement
            end_time = time.time()  # Temps après l'entraînement

            training_time = end_time - start_time  # Calculer le temps d'entraînement
            memory_used = max(0, mem_after - mem_before)  # S'assurer que la mémoire utilisée n'est pas négative

            # Récupérer le meilleur modèle et ses paramètres
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Faire des prédictions sur l'ensemble de test
            y_pred = best_model.predict(X_test)

            # Calculer les probabilités pour AUC-ROC et AUC-PR si applicable
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_proba = best_model.decision_function(X_test)
            else:
                y_proba = None

            # Calculer les métriques de performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Calculer AUC-ROC et AUC-PR si y_proba est disponible
            auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            auc_pr = average_precision_score(y_test, y_proba) if y_proba is not None else None

            # Stocker les résultats dans la liste
            results.append({
                'group_name': group_name,
                'dataset_name': dataset_name,
                'train_size': train_size,  # Taille de l'ensemble d'entraînement
                'test_size': test_size,  # Taille de l'ensemble de test
                'test_positives': num_positives,  # Nombre de positifs dans y_test
                'best_params': best_params,  # Meilleurs paramètres sélectionnés
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'training_time': training_time,  # Temps d'entraînement
                'memory_used': memory_used  # Mémoire utilisée
            })

            # Stocker le modèle ajusté
            best_models[dataset_name] = best_model

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)

    # Calculer les moyennes par groupe pour toutes les métriques
    group_means = results_df.groupby('group_name').mean(numeric_only=True).add_prefix('group_mean_')

    # Fusionner les moyennes avec les résultats
    results_df = results_df.merge(group_means, on='group_name', how='left')

    # Retourner les résultats et les modèles ajustés
    return results_df, best_models




def summarize_model_perfs_Group1_Group2(results_dict):
    """
    Cette fonction prend un dictionnaire de résultats de modèles et retourne un DataFrame
    avec deux colonnes ('Group 1' et 'Group 2') et des lignes représentant les moyennes
    des métriques (accuracy, precision, recall, F1-Score, AUC-ROC, AUC-PR, temps d'exécution, mémoire utilisée)
    sur tous les modèles pour chaque groupe.

    :param results_dict: Dictionnaire avec comme clés les noms des modèles et comme valeurs les DataFrames des résultats.
    :return: DataFrame avec deux colonnes 'Group 1' et 'Group 2', et les lignes pour chaque métrique (accuracy, precision, etc.).
    """
    # Initialiser des dictionnaires pour stocker les métriques pour chaque groupe
    group1_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': [], 'auc_pr': [], 'training_time': [], 'memory_used': []}
    group2_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': [], 'auc_pr': [], 'training_time': [], 'memory_used': []}

    # Parcourir chaque modèle et récupérer les résultats pour les deux groupes
    for model_name, result_df in results_dict.items():
        # Extraire les moyennes pour Group 1
        group1_mean = result_df[result_df['group_name'] == 'group1'].mean(numeric_only=True)
        # Extraire les moyennes pour Group 2
        group2_mean = result_df[result_df['group_name'] == 'group2'].mean(numeric_only=True)

        # Ajouter les métriques moyennes dans les dictionnaires respectifs
        group1_metrics['accuracy'].append(group1_mean['accuracy'])
        group1_metrics['precision'].append(group1_mean['precision'])
        group1_metrics['recall'].append(group1_mean['recall'])
        group1_metrics['f1_score'].append(group1_mean['f1_score'])
        group1_metrics['auc_roc'].append(group1_mean['auc_roc'])
        group1_metrics['auc_pr'].append(group1_mean['auc_pr'])  # AUC des classes positives
        group1_metrics['training_time'].append(group1_mean['training_time'])  # Temps d'exécution
        group1_metrics['memory_used'].append(group1_mean['memory_used'])  # Mémoire utilisée

        group2_metrics['accuracy'].append(group2_mean['accuracy'])
        group2_metrics['precision'].append(group2_mean['precision'])
        group2_metrics['recall'].append(group2_mean['recall'])
        group2_metrics['f1_score'].append(group2_mean['f1_score'])
        group2_metrics['auc_roc'].append(group2_mean['auc_roc'])
        group2_metrics['auc_pr'].append(group2_mean['auc_pr'])  # AUC des classes positives
        group2_metrics['training_time'].append(group2_mean['training_time'])  # Temps d'exécution
        group2_metrics['memory_used'].append(group2_mean['memory_used'])  # Mémoire utilisée

    # Calculer les moyennes globales pour chaque métrique
    summary_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR', 'Training Time', 'Memory Used'],
        'Group 1': [
            np.mean(group1_metrics['accuracy']),
            np.mean(group1_metrics['precision']),
            np.mean(group1_metrics['recall']),
            np.mean(group1_metrics['f1_score']),
            np.mean(group1_metrics['auc_roc']),
            np.mean(group1_metrics['auc_pr']),
            np.mean(group1_metrics['training_time']),
            np.mean(group1_metrics['memory_used'])
        ],
        'Group 2': [
            np.mean(group2_metrics['accuracy']),
            np.mean(group2_metrics['precision']),
            np.mean(group2_metrics['recall']),
            np.mean(group2_metrics['f1_score']),
            np.mean(group2_metrics['auc_roc']),
            np.mean(group2_metrics['auc_pr']),
            np.mean(group2_metrics['training_time']),
            np.mean(group2_metrics['memory_used'])
        ]
    }

    # Créer un DataFrame à partir des moyennes calculées
    summary_df = pd.DataFrame(summary_data)

    return summary_df

def basic_fintune_and_evaluate_CV_SMOTE(train_test_data, classifier, param_name, param_values, cv=5, scoring='recall', smote_strategy=1.0):
    """
    Évalue chaque dataset dans train_test_data en fine-tunant un paramètre du classificateur à l'aide de GridSearchCV.
    Intègre SMOTE pour équilibrer les classes à 50/50 entre positives et négatives.
    Utilise un pipeline avec SMOTE, RobustScaler pour gérer les outliers et un suivi du temps d'entraînement pour estimer la complexité temporelle.

    Paramètres :
    - train_test_data : Dictionnaire contenant les datasets (X_train, X_test, y_train, y_test) par groupe et nom de dataset.
    - classifier : Le modèle de classification à fine-tuner.
    - param_name : Nom du paramètre à fine-tuner.
    - param_values : Liste des valeurs pour le paramètre à tester.
    - cv : Nombre de folds pour la validation croisée (par défaut 5).
    - scoring : Métrique utilisée pour l'évaluation (par défaut 'accuracy').

    Retourne :
    - results_df : DataFrame contenant les résultats des métriques pour chaque dataset.
    - best_models : Dictionnaire contenant les modèles ajustés pour chaque dataset.
    """

    results = []
    best_models = {}  # Dictionnaire pour stocker les modèles ajustés

    # Parcourir les groupes de datasets
    for group_name in train_test_data:
        # Parcourir chaque dataset dans le groupe
        for dataset_name, data in train_test_data[group_name].items():
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

            # Taille des ensembles de données
            train_size = len(X_train)
            test_size = len(X_test)
            num_positives = sum(y_test == 1)  # Compte des classes positives dans y_test

            # Créer un pipeline avec SMOTE, RobustScaler et le classificateur
            pipeline = ImbPipeline([
                ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=0, n_jobs=-1)),  # SMOTE pour équilibrer les classes à 50/50
                ('scaler', RobustScaler()),  # normaliser et gérer les outliers avec RobustScaler
                ('classifier', classifier)  # Le classificateur à fine-tuner
            ])

            # Configurer GridSearchCV pour la recherche du meilleur paramètre
            param_grid = {f'classifier__{param_name}': param_values}  # Le format pour GridSearch avec pipeline
            # Stratifier les folds, pour sauvegarder les proportions de classe
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring=scoring, n_jobs=-1)

            # Suivi du temps d'entraînement
            start_time = time.time()  # Temps avant l'entraînement
            mem_before = memory_usage()[0]  # Utilisation de la mémoire avant l'entraînement

            grid_search.fit(X_train, y_train)  # Entraîner le modèle avec la recherche de grille

            mem_after = memory_usage()[0]  # Utilisation de la mémoire après l'entraînement
            end_time = time.time()  # Temps après l'entraînement

            training_time = end_time - start_time  # Calculer le temps d'entraînement
            memory_used = max(0, mem_after - mem_before)  # S'assurer que la mémoire utilisée n'est pas négative

            # Récupérer le meilleur paramètre et le meilleur modèle
            best_param = grid_search.best_params_[f'classifier__{param_name}']
            best_model = grid_search.best_estimator_

            # Faire des prédictions sur l'ensemble de test
            y_pred = best_model.predict(X_test)

            # Calculer les probabilités pour AUC-ROC et AUC-PR si applicable
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_proba = best_model.decision_function(X_test)
            else:
                y_proba = None

            # Calculer les métriques de performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Calculer AUC-ROC et AUC-PR si y_proba est disponible
            auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            auc_pr = average_precision_score(y_test, y_proba) if y_proba is not None else None

            # Stocker les résultats dans la liste
            results.append({
                'group_name': group_name,
                'dataset_name': dataset_name,
                'train_size': train_size,  # Taille de l'ensemble d'entraînement
                'test_size': test_size,  # Taille de l'ensemble de test
                'test_positives': num_positives,  # Nombre de positifs dans y_test
                'best_param': best_param,  # Meilleur paramètre sélectionné
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'training_time': training_time,  # Temps d'entraînement
                'memory_used': memory_used  # Mémoire utilisée
            })

            # Stocker le modèle ajusté
            best_models[dataset_name] = best_model

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)

    # Calculer les moyennes par groupe pour toutes les métriques
    group_means = results_df.groupby('group_name').mean(numeric_only=True).add_prefix('group_mean_')

    # Fusionner les moyennes avec les résultats
    results_df = results_df.merge(group_means, on='group_name', how='left')

    # Retourner les résultats et les modèles ajustés
    return results_df, best_models


def ensemble_fine_tune_and_evaluate_CV_SMOTE(train_test_data, classifier, param_grid, cv=5, scoring='recall', smote_strategy=1.0):
    """
    Évalue chaque dataset dans train_test_data en fine-tunant plusieurs paramètres du classificateur
    en utilisant GridSearchCV. Intègre SMOTE pour équilibrer les classes à 50/50 entre positives et négatives.
    Conçu pour les méthodes ensemblistes (Bagging, Boosting, RandomForest, etc.) sans pondération des poids,
    avec un suivi du temps d'entraînement et de l'utilisation mémoire.

    Paramètres :
    - train_test_data : Dictionnaire contenant les datasets (X_train, X_test, y_train, y_test) par groupe et nom de dataset.
    - classifier : Le modèle d'ensemble à fine-tuner.
    - param_grid : Grille de paramètres à fine-tuner sous forme de dictionnaire (GridSearchCV).
    - cv : Nombre de folds pour la validation croisée (par défaut 5).
    - scoring : Métrique utilisée pour l'évaluation (par défaut 'accuracy').

    Retourne :
    - results_df : DataFrame contenant les résultats des métriques pour chaque dataset.
    - best_models : Dictionnaire contenant les modèles ajustés pour chaque dataset.
    """

    results = []
    best_models = {}  # Dictionnaire pour stocker les modèles ajustés

    # Parcourir les groupes de datasets
    for group_name in train_test_data:
        # Parcourir chaque dataset dans le groupe
        for dataset_name, data in train_test_data[group_name].items():
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

            # Taille des ensembles de données
            train_size = len(X_train)
            test_size = len(X_test)
            num_positives = sum(y_test == 1)  # Compte des classes positives dans y_test

            # Créer un pipeline avec SMOTE, RobustScaler et le classificateur
            pipeline = ImbPipeline([
                ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=0)),  # SMOTE pour équilibrer les classes à 50/50
                ('scaler', RobustScaler()),  # Gérer les outliers avec RobustScaler
                ('classifier', classifier)   # Le classificateur d'ensemble à fine-tuner
            ])

            # Stratifier les folds, pour sauvegarder les proportions de classe
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

            # Configurer GridSearchCV pour la recherche du meilleur paramètre
            grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring=scoring, n_jobs=-1)

            # Suivi du temps d'entraînement
            start_time = time.time()  # Temps avant l'entraînement
            mem_before = memory_usage()[0]  # Utilisation de la mémoire avant l'entraînement

            # Entraîner le modèle avec la recherche de grille
            grid_search.fit(X_train, y_train)

            mem_after = memory_usage()[0]  # Utilisation de la mémoire après l'entraînement
            end_time = time.time()  # Temps après l'entraînement

            training_time = end_time - start_time  # Calculer le temps d'entraînement
            memory_used = max(0, mem_after - mem_before)  # S'assurer que la mémoire utilisée n'est pas négative

            # Récupérer le meilleur modèle et ses paramètres
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Faire des prédictions sur l'ensemble de test
            y_pred = best_model.predict(X_test)

            # Calculer les probabilités pour AUC-ROC et AUC-PR si applicable
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_proba = best_model.decision_function(X_test)
            else:
                y_proba = None

            # Calculer les métriques de performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Calculer AUC-ROC et AUC-PR si y_proba est disponible
            auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            auc_pr = average_precision_score(y_test, y_proba) if y_proba is not None else None

            # Stocker les résultats dans la liste
            results.append({
                'group_name': group_name,
                'dataset_name': dataset_name,
                'train_size': train_size,  # Taille de l'ensemble d'entraînement
                'test_size': test_size,    # Taille de l'ensemble de test
                'test_positives': num_positives,  # Nombre de positifs dans y_test
                'best_params': best_params,  # Meilleurs paramètres sélectionnés
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'training_time': training_time,  # Temps d'entraînement
                'memory_used': memory_used  # Mémoire utilisée
            })

            # Stocker le modèle ajusté
            best_models[dataset_name] = best_model

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)

    # Calculer les moyennes par groupe pour toutes les métriques
    group_means = results_df.groupby('group_name').mean(numeric_only=True).add_prefix('group_mean_')

    # Fusionner les moyennes avec les résultats
    results_df = results_df.merge(group_means, on='group_name', how='left')

    # Retourner les résultats et les modèles ajustés
    return results_df, best_models

def ensemble_fine_tune_and_evaluate_CV_XGB(train_test_data, classifier, param_grid, cv=5, scoring='recall'):
    """
    Évalue chaque dataset dans train_test_data en fine-tunant plusieurs paramètres du classificateur
    en utilisant GridSearchCV. Conçu pour les méthodes ensemblistes (Bagging, Boosting, RandomForest, etc.)
    avec un suivi du temps d'entraînement et de l'utilisation mémoire. Ajout dynamique de scale_pos_weight.

    Paramètres :
    - train_test_data : Dictionnaire contenant les datasets (X_train, X_test, y_train, y_test) par groupe et nom de dataset.
    - classifier : Le modèle d'ensemble à fine-tuner.
    - param_grid : Grille de paramètres à fine-tuner sous forme de dictionnaire (GridSearchCV).
    - cv : Nombre de folds pour la validation croisée (par défaut 5).
    - scoring : Métrique utilisée pour l'évaluation (par défaut 'recall').

    Retourne :
    - results_df : DataFrame contenant les résultats des métriques pour chaque dataset.
    - best_models : Dictionnaire contenant les modèles ajustés pour chaque dataset.
    """

    results = []
    best_models = {}  # Dictionnaire pour stocker les modèles ajustés

    # Parcourir les groupes de datasets
    for group_name in train_test_data:
        # Parcourir chaque dataset dans le groupe
        for dataset_name, data in train_test_data[group_name].items():
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

            # Calculer le ratio de scale_pos_weight (nombre de négatifs / nombre de positifs)
            num_positives = sum(y_train == 1)
            num_negatives = sum(y_train == 0)
            scale_pos_weight = num_negatives / num_positives if num_positives > 0 else 1

            # Mettre à jour la grille de paramètres pour inclure scale_pos_weight
            param_grid['classifier__scale_pos_weight'] = [scale_pos_weight]

            # Créer un pipeline avec RobustScaler et le classificateur
            pipeline = Pipeline([
                ('scaler', RobustScaler()),  # Gérer les outliers avec RobustScaler
                ('classifier', classifier)  # Le classificateur d'ensemble à fine-tuner
            ])

            # Stratifier les folds, pour sauvegarder les proportions de classe
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

            # Configurer GridSearchCV pour la recherche du meilleur paramètre
            grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring=scoring, n_jobs=-1)

            # Suivi du temps d'entraînement
            start_time = time.time()  # Temps avant l'entraînement
            mem_before = memory_usage()[0]  # Utilisation de la mémoire avant l'entraînement

            # Entraîner le modèle avec la recherche de grille
            grid_search.fit(X_train, y_train)

            mem_after = memory_usage()[0]  # Utilisation de la mémoire après l'entraînement
            end_time = time.time()  # Temps après l'entraînement

            training_time = end_time - start_time  # Calculer le temps d'entraînement
            memory_used = max(0, mem_after - mem_before)  # S'assurer que la mémoire utilisée n'est pas négative

            # Récupérer le meilleur modèle et ses paramètres
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Faire des prédictions sur l'ensemble de test
            y_pred = best_model.predict(X_test)

            # Calculer les probabilités pour AUC-ROC et AUC-PR si applicable
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_proba = best_model.decision_function(X_test)
            else:
                y_proba = None

            # Calculer les métriques de performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Calculer AUC-ROC et AUC-PR si y_proba est disponible
            auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            auc_pr = average_precision_score(y_test, y_proba) if y_proba is not None else None

            # Stocker les résultats dans la liste
            results.append({
                'group_name': group_name,
                'dataset_name': dataset_name,
                'train_size': len(X_train),  # Taille de l'ensemble d'entraînement
                'test_size': len(X_test),  # Taille de l'ensemble de test
                'test_positives': sum(y_test == 1),  # Nombre de positifs dans y_test
                'best_params': best_params,  # Meilleurs paramètres sélectionnés
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'training_time': training_time,  # Temps d'entraînement
                'memory_used': memory_used  # Mémoire utilisée
            })

            # Stocker le modèle ajusté
            best_models[dataset_name] = best_model

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)

    # Calculer les moyennes par groupe pour toutes les métriques
    group_means = results_df.groupby('group_name').mean(numeric_only=True).add_prefix('group_mean_')

    # Fusionner les moyennes avec les résultats
    results_df = results_df.merge(group_means, on='group_name', how='left')

    # Retourner les résultats et les modèles ajustés
    return results_df, best_models


def collect_metric_columns(global_results, metric):
    metric_data = {}

    # Parcourir chaque famille de test dans global_results (e.g., 'basic', 'ensemble', 'basic_smoted', etc.)
    for family_name, family_results in global_results.items():
        # Parcourir chaque méthode dans la famille de tests (e.g., 'Logistic Regression', 'SVM', etc.)
        for method_name, method_results in family_results.items():
            # Pour chaque dataset, extraire la colonne correspondant à la métrique
            for idx, row in method_results.iterrows():
                dataset_name = row['dataset_name']  # Utilise 'dataset_name' comme clé
                if dataset_name not in metric_data:
                    metric_data[dataset_name] = {}  # Initialiser un dictionnaire pour chaque dataset

                # Créer un nom de colonne dynamique, par exemple 'F1-score_basic_SVM'
                column_name = f"{metric}_{family_name}_{method_name}"

                # Extraire la métrique directement par le nom de la colonne
                if metric in row:
                    metric_data[dataset_name][column_name] = row[metric]  # Ajouter la métrique au dictionnaire
                else:
                    metric_data[dataset_name][column_name] = None  # Si la métrique n'est pas présente, mettre None

    # Convertir les données collectées en DataFrame
    metric_df = pd.DataFrame(metric_data).T  # Transposer pour avoir les datasets en lignes et méthodes en colonnes

    # Afficher le DataFrame
    #print(f"Tableau des résultats pour la métrique '{metric}':")
    #display(metric_df)

    return metric_df


# Fonction pour visualiser la heatmap des résultats
def visualize_metric_heatmap(metric_df, metric):
    # Visualiser sous forme de heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(metric_df, annot=True, cmap="coolwarm", cbar=True, linewidths=0.5)
    plt.title(f"Heatmap de la métrique '{metric}' pour chaque méthode et famille de test")
    plt.show()




# Fonctions fournies par l'énoncé

def loadCsv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    (n, d) = data.shape
    return data, n, d

# Fonction oneHotEncodeColums modifiée pour éviter la colinéarité lors de l'encodage des variables catégorielles

def oneHotEncodeColumns(data, columnsCategories):
    dataCategories = data[:, columnsCategories]
    dataEncoded = OneHotEncoder(sparse_output=False, drop="first").fit_transform(dataCategories)
    columnsNumerical = []
    for i in range(data.shape[1]):
        if i not in columnsCategories:
            columnsNumerical.append(i)
    dataNumerical = data[:, columnsNumerical]
    return np.hstack((dataNumerical, dataEncoded)).astype(float)



# Fonction data_recovery modifiée pour éviter la colinéarité lors de l'encodage des variables catégorielles

def data_recovery(dataset):
    if dataset in ['abalone8', 'abalone17', 'abalone20']:
        data = pd.read_csv("datasets/abalone.data", header=None)
        data = pd.get_dummies(data, dtype=float, drop_first=True)
        if dataset in ['abalone8']:
            y = np.array([1 if elt == 8 else 0 for elt in data[8]])
        elif dataset in ['abalone17']:
            y = np.array([1 if elt == 17 else 0 for elt in data[8]])
        elif dataset in ['abalone20']:
            y = np.array([1 if elt == 20 else 0 for elt in data[8]])
        X = np.array(data.drop([8], axis=1))
    elif dataset in ['autompg']:
        data = pd.read_csv("datasets/auto-mpg.data", header=None, sep=r'\s+')
        data = data.replace('?', np.nan)
        data = data.dropna()
        data = data.drop([8], axis=1)
        data = data.astype(float)
        y = np.array([1 if elt in [2, 3] else 0 for elt in data[7]])
        X = np.array(data.drop([7], axis=1))
    elif dataset in ['australian']:
        data, n, d = loadCsv('datasets/australian.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1].astype(int)
        y[y != 1] = 0
    elif dataset in ['balance']:
        data = pd.read_csv("datasets/balance-scale.data", header=None)
        y = np.array([1 if elt in ['L'] else 0 for elt in data[0]])
        X = np.array(data.drop([0], axis=1))
    elif dataset in ['bankmarketing']:
        data, n, d = loadCsv('datasets/bankmarketing.csv')
        X = data[:, np.arange(0, d-1)]
        X = oneHotEncodeColumns(X, [1, 2, 3, 4, 6, 7, 8, 10, 15])
        y = data[:, d-1]
        y[y == "no"] = "0"
        y[y == "yes"] = "1"
        y = y.astype(int)
    elif dataset in ['bupa']:
        data, n, d = loadCsv('datasets/bupa.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1].astype(int)
        y[y != 1] = 0
    elif dataset in ['german']:
        data = pd.read_csv("datasets/german.data-numeric", header=None,
                           sep=r'\s+')
        y = np.array([1 if elt == 2 else 0 for elt in data[24]])
        X = np.array(data.drop([24], axis=1))
    elif dataset in ['glass']:
        data = pd.read_csv("datasets/glass.data", header=None, index_col=0)
        y = np.array([1 if elt == 1 else 0 for elt in data[10]])
        X = np.array(data.drop([10], axis=1))
    elif dataset in ['hayes']:
        data = pd.read_csv("datasets/hayes-roth.data", header=None)
        y = np.array([1 if elt in [3] else 0 for elt in data[5]])
        X = np.array(data.drop([0, 5], axis=1))
    elif dataset in ['heart']:
        data, n, d = loadCsv('datasets/heart.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y = y.astype(int)
        y[y != 2] = 0
        y[y == 2] = 1
    elif dataset in ['iono']:
        data = pd.read_csv("datasets/ionosphere.data", header=None)
        y = np.array([1 if elt in ['b'] else 0 for elt in data[34]])
        X = np.array(data.drop([34], axis=1))
    elif dataset in ['libras']:
        data = pd.read_csv("datasets/movement_libras.data", header=None)
        y = np.array([1 if elt in [1] else 0 for elt in data[90]])
        X = np.array(data.drop([90], axis=1))
    elif dataset == "newthyroid":
        data, n, d = loadCsv('datasets/newthyroid.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1].astype(int)
        y[y < 2] = 0
        y[y >= 2] = 1
    elif dataset in ['pageblocks']:
        data = pd.read_csv("datasets/page-blocks.data", header=None,
                           sep=r'\s+')
        y = np.array([1 if elt in [2, 3, 4, 5] else 0 for elt in data[10]])
        X = np.array(data.drop([10], axis=1))
    elif dataset in ['pima']:
        data, n, d = loadCsv('datasets/pima-indians-diabetes.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y[y != '1'] = '0'
        y = y.astype(int)
    elif dataset in ['satimage']:
        data, n, d = loadCsv('datasets/satimage.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y = y.astype(int)
        y[y != 4] = 0
        y[y == 4] = 1
    elif dataset in ['segmentation']:
        data, n, d = loadCsv('datasets/segmentation.data')
        X = data[:, np.arange(1, d)].astype(float)
        y = data[:, 0]
        y[y == "WINDOW"] = '1'
        y[y != '1'] = '0'
        y = y.astype(int)
    elif dataset == "sonar":
        data, n, d = loadCsv('datasets/sonar.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y[y != 'R'] = '0'
        y[y == 'R'] = '1'
        y = y.astype(int)
    elif dataset == "spambase":
        data, n, d = loadCsv('datasets/spambase.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1].astype(int)
        y[y != 1] = 0
    elif dataset == "splice":
        data, n, d = loadCsv('datasets/splice.data')
        X = data[:, np.arange(1, d)].astype(float)
        y = data[:, 0].astype(int)
        y[y == 1] = 2
        y[y == -1] = 1
        y[y == 2] = 0
    elif dataset in ['vehicle']:
        data, n, d = loadCsv('datasets/vehicle.data')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y[y != "van"] = '0'
        y[y == "van"] = '1'
        y = y.astype(int)
    elif dataset in ['wdbc']:
        data, n, d = loadCsv('datasets/wdbc.dat')
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y[y != 'M'] = '0'
        y[y == 'M'] = '1'
        y = y.astype(int)
    elif dataset in ['wine']:
        data = pd.read_csv("datasets/wine.data", header=None)
        y = np.array([1 if elt == 1 else 0 for elt in data[0]])
        X = np.array(data.drop([0], axis=1))
    elif dataset in ['wine4']:
        data = pd.read_csv("datasets/winequality-red.csv", sep=';')
        y = np.array([1 if elt in [4] else 0 for elt in data.quality])
        X = np.array(data.drop(["quality"], axis=1))
    elif dataset in ['yeast3', 'yeast6']:
        data = pd.read_csv("datasets/yeast.data", header=None, sep=r'\s+')
        data = data.drop([0], axis=1)
        if dataset == 'yeast3':
            y = np.array([1 if elt == 'ME3' else 0 for elt in data[9]])
        elif dataset == 'yeast6':
            y = np.array([1 if elt == 'EXC' else 0 for elt in data[9]])
        X = np.array(data.drop([9], axis=1))
    return X, y
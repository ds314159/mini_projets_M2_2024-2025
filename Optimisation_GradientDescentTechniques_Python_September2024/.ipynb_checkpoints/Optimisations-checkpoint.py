import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from Outils import *
np.random.seed(1337)




# Implémantation de descente de gradient simple:
def gradient_descent(X, y, w_random, learning_rate, num_iterations):
    """
    Implémente l'algorithme de descente de gradient pour la régression logistique.

    Args:
    X : matrice des caractéristiques (m échantillons par n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage pour la mise à jour des poids
    num_iterations : nombre d'itérations pour l'algorithme

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à chaque itération
    """
    m = len(y)  # Nombre d'échantillons
    costs = []  # Liste pour stocker l'évolution du coût
    w = w_random.copy() # pour ne pas altérer les poids d'entrée s'il faut les réutiliser

    for i in range(num_iterations):
        # Calcul des prédictions avec les poids actuels
        y_pred = predict(X, w)

        # Calcul du gradient pour les poids (sauf le biais)
        dw = ( 1 /m) * np.dot(X.T, (y_pred - y))
        # Calcul du gradient pour le biais
        db = ( 1 /m) * np.sum(y_pred - y)

        # Mise à jour des poids (sauf le biais) avec la règle de descente de gradient
        w[1:] -= learning_rate * dw
        # Mise à jour du biais
        w[0] -= learning_rate * db

        # Calcul et stockage du coût pour cette itération
        cost = cost_function(X, y, w)
        costs.append(cost)
    return w, costs



# Implémentation de la descente de gradient stochastique, extensible en changeant la taille du batch:
def stochastic_gradient_descent(X, y, w_random, learning_rate, num_epochs, batch_size=1):
    """
    Implémente l'algorithme de descente de gradient stochastique pour la régression logistique.

    Args:
    X : matrice des caractéristiques (m échantillons x n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage pour la mise à jour des poids
    num_epochs : nombre d'époques (passages complets sur les données)
    batch_size : taille des mini-batchs (1 pour SGD pur, >1 pour mini-batch SGD)

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à intervalles réguliers
    """
    m = len(y)  # Nombre total d'échantillons
    costs = []  # Liste pour stocker l'évolution du coût
    w = w_random.copy() # pour ne pas altérer les poids d'entrée s'il faut les réutiliser

    for epoch in range(num_epochs):
        # Mélange aléatoire des indices des échantillons
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Parcours des mini-batchs
        for i in range(0, m, batch_size):
            # Sélection du mini-batch courant
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Calcul des prédictions pour le mini-batch
            y_pred = predict(X_batch, w)

            # Calcul du gradient pour les poids (sauf le biais) sur le mini-batch
            dw = (1/batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            # Calcul du gradient pour le biais sur le mini-batch
            db = (1/batch_size) * np.sum(y_pred - y_batch)

            # Mise à jour des poids (sauf le biais)
            w[1:] -= learning_rate * dw
            # Mise à jour du biais
            w[0] -= learning_rate * db

            # Calcul et stockage du coût
            cost = cost_function(X, y, w)
            costs.append(cost)

    return w, costs


# Implémentation de la descente de gradient avec Momentum:
def gradient_descent_momentum(X, y, w_random, learning_rate, num_iterations, momentum=0.9):
    """
    Implémente l'algorithme de descente de gradient avec Momentum pour la régression logistique.

    Args:
    X : matrice des caractéristiques (m échantillons x n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage pour la mise à jour des poids
    num_iterations : nombre d'itérations pour l'algorithme
    momentum : coefficient de momentum (par défaut 0.9)

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à chaque itération
    """
    m = len(y)  # Nombre d'échantillons
    costs = []  # Liste pour stocker l'évolution du coût
    w = w_random.copy()  # pour ne pas altérer les poids d'entrée s'il faut les réutiliser
    velocity = np.zeros_like(w)  # Initialisation de la vélocité

    for i in range(num_iterations):
        # Calcul des prédictions avec les poids actuels
        y_pred = predict(X, w)

        # Calcul du gradient
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Mise à jour de la vélocité
        velocity[1:] = momentum * velocity[1:] + learning_rate * dw
        velocity[0] = momentum * velocity[0] + learning_rate * db

        # Mise à jour des poids avec la vélocité
        w[1:] -= velocity[1:]
        w[0] -= velocity[0]

        # Calcul et stockage du coût pour cette itération
        cost = cost_function(X, y, w)
        costs.append(cost)
    return w, costs


# Implémentation de la descente de gradient avec la méthode de Nesterov:
def gradient_descent_nesterov(X, y, w_random, learning_rate, num_iterations, momentum=0.9):
    """
    Implémente l'algorithme de descente de gradient avec la méthode de Nesterov pour la régression logistique.

    Args:
    X : matrice des caractéristiques (m échantillons x n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage pour la mise à jour des poids
    num_iterations : nombre d'itérations pour l'algorithme
    momentum : coefficient de momentum (par défaut 0.9)

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à chaque itération
    """

    m, n = X.shape
    costs = []
    w = w_random.copy()  # pour ne pas altérer les poids d'entrée s'il faut les réutiliser
    velocity = np.zeros_like(w)

    for i in range(num_iterations):
        # Calcul de la position anticipée
        w_ahead = w + momentum * velocity

        # Calcul du gradient à la position anticipée
        y_pred_ahead = predict(X, w_ahead)
        gradient_ahead = np.zeros_like(w)
        gradient_ahead[1:] = (1 / m) * np.dot(X.T, (y_pred_ahead - y))
        gradient_ahead[0] = (1 / m) * np.sum(y_pred_ahead - y)

        # Mise à jour de la vélocité avec Nesterov (utilise w_ahead pour le gradient)
        velocity = momentum * velocity - learning_rate * gradient_ahead

        # Mise à jour des poids (en appliquant la vélocité)
        w += velocity

        # Calcul et stockage du coût
        cost = cost_function(X, y, w)
        costs.append(cost)

    return w, costs


# Implémentation de la descente de gradient avec AdaGrad:
def gradient_descent_adagrad(X, y, w_random, learning_rate, num_iterations, epsilon=1e-8):
    """
    Implémente l'algorithme de descente de gradient avec AdaGrad pour la régression logistique.

    Args:
    X : matrice des caractéristiques (m échantillons x n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage initial
    num_iterations : nombre d'itérations pour l'algorithme
    epsilon : petit terme pour éviter la division par zéro

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à chaque itération
    """
    m = len(y)  # Nombre d'échantillons
    costs = []  # Liste pour stocker l'évolution du coût
    w = w_random.copy()  # pour ne pas altérer les poids d'entrée s'il faut les réutiliser
    G = np.zeros_like(w)  # Initialisation de l'accumulateur de gradients au carré

    for i in range(num_iterations):
        # Calcul des prédictions avec les poids actuels
        y_pred = predict(X, w)

        # Calcul du gradient
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Accumulation des gradients au carré
        G[1:] += dw ** 2
        G[0] += db ** 2

        # Mise à jour des poids
        w[1:] -= (learning_rate / (np.sqrt(G[1:] + epsilon))) * dw
        w[0] -= (learning_rate / (np.sqrt(G[0] + epsilon))) * db

        # Calcul et stockage du coût pour cette itération
        cost = cost_function(X, y, w)
        costs.append(cost)

    return w, costs


# Implémentation de la descente de gradient avec RMSprop:
def gradient_descent_rmsprop(X, y, w_random, learning_rate, num_iterations, beta=0.9, epsilon=1e-8):
    """
    Implémente l'algorithme de descente de gradient avec RMSprop pour la régression logistique.

    Args:
    X : matrice des caractéristiques (m échantillons x n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage
    num_iterations : nombre d'itérations pour l'algorithme
    beta : facteur de décroissance pour la moyenne mobile (par défaut 0.9)
    epsilon : petit terme pour éviter la division par zéro

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à chaque itération
    """
    m = len(y)  # Nombre d'échantillons
    costs = []  # Liste pour stocker l'évolution du coût
    w = w_random.copy()  # pour ne pas altérer les poids d'entrée s'il faut les réutiliser
    v = np.zeros_like(w)  # Initialisation de la moyenne mobile des gradients au carré

    for i in range(num_iterations):
        # Calcul des prédictions avec les poids actuels
        y_pred = predict(X, w)

        # Calcul du gradient
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Mise à jour de la moyenne mobile des gradients au carré
        v[1:] = beta * v[1:] + (1 - beta) * dw ** 2
        v[0] = beta * v[0] + (1 - beta) * db ** 2

        # Mise à jour des poids
        w[1:] -= (learning_rate / (np.sqrt(v[1:] + epsilon))) * dw
        w[0] -= (learning_rate / (np.sqrt(v[0] + epsilon))) * db

        # Calcul et stockage du coût pour cette itération
        cost = cost_function(X, y, w)
        costs.append(cost)

    return w, costs


# Implémentation de la descente de gradient avec Adam:
def gradient_descent_adam(X, y, w_random, learning_rate, num_iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Implémente l'algorithme de descente de gradient avec Adam pour la régression logistique.

    Args:
    X : matrice des caractéristiques (m échantillons x n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage
    num_iterations : nombre d'itérations pour l'algorithme
    beta1 : facteur de décroissance pour l'estimation du premier moment (par défaut 0.9)
    beta2 : facteur de décroissance pour l'estimation du second moment (par défaut 0.999)
    epsilon : petit terme pour éviter la division par zéro

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à chaque itération
    """
    m = len(y)  # Nombre d'échantillons
    costs = []  # Liste pour stocker l'évolution du coût
    w = w_random.copy()  # pour ne pas altérer les poids d'entrée s'il faut les réutiliser
    m_t = np.zeros_like(w)  # Initialisation de l'estimation du premier moment
    v_t = np.zeros_like(w)  # Initialisation de l'estimation du second moment
    t = 0  # Initialisation du compteur de temps

    for i in range(num_iterations):
        t += 1  # Incrémentation du compteur de temps

        # Calcul des prédictions avec les poids actuels
        y_pred = predict(X, w)

        # Calcul du gradient
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Mise à jour des estimations des moments
        m_t[1:] = beta1 * m_t[1:] + (1 - beta1) * dw
        m_t[0] = beta1 * m_t[0] + (1 - beta1) * db
        v_t[1:] = beta2 * v_t[1:] + (1 - beta2) * dw ** 2
        v_t[0] = beta2 * v_t[0] + (1 - beta2) * db ** 2

        # Correction du biais
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)

        # Mise à jour des poids
        w[1:] -= (learning_rate / (np.sqrt(v_t_hat[1:]) + epsilon)) * m_t_hat[1:]
        w[0] -= (learning_rate / (np.sqrt(v_t_hat[0]) + epsilon)) * m_t_hat[0]

        # Calcul et stockage du coût pour cette itération
        cost = cost_function(X, y, w)
        costs.append(cost)

    return w, costs


# Descente de gradient avec une pénalisation Ridge
def gradient_descent_ridge(X, y, w_random, learning_rate, num_iterations, lambda_reg):
    """
    Implémente l'algorithme de descente de gradient pour la régression logistique avec régularisation Ridge (L2).

    Args:
    X : matrice des caractéristiques (m échantillons x n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w_random : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage
    num_iterations : nombre d'itérations pour l'algorithme
    lambda_reg : coefficient de régularisation L2

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à chaque itération
    """
    m = len(y)  # Nombre d'échantillons
    costs = []  # Liste pour stocker l'évolution du coût
    w = w_random.copy()  # Pour ne pas altérer les poids d'entrée s'il faut les réutiliser

    for i in range(num_iterations):
        # Calcul des prédictions avec les poids actuels
        y_pred = predict(X, w)

        # Calcul du gradient pour les poids (sauf le biais), avec régularisation L2
        dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (lambda_reg / m) * w[1:]
        # Calcul du gradient pour le biais (le biais n'est pas régularisé)
        db = (1 / m) * np.sum(y_pred - y)

        # Mise à jour des poids (sauf le biais)
        w[1:] -= learning_rate * dw
        # Mise à jour du biais
        w[0] -= learning_rate * db

        # Calcul et stockage du coût, en incluant le terme de régularisation
        cost = cost_function(X, y, w) + (lambda_reg / (2 * m)) * np.sum(w[1:] ** 2)
        costs.append(cost)

    return w, costs

# Descente de gradient avec une pénalisation Lasso
def gradient_descent_lasso(X, y, w_random, learning_rate, num_iterations, lambda_reg):
    """
    Implémente l'algorithme de descente de gradient pour la régression logistique avec régularisation Lasso (L1).

    Args:
    X : matrice des caractéristiques (m échantillons x n caractéristiques)
    y : vecteur des étiquettes (m échantillons)
    w_random : vecteur initial des poids (n+1 dimensions, incluant le biais)
    learning_rate : taux d'apprentissage
    num_iterations : nombre d'itérations pour l'algorithme
    lambda_reg : coefficient de régularisation L1

    Returns:
    w : vecteur final des poids optimisés
    costs : liste des coûts à chaque itération
    """
    m = len(y)  # Nombre d'échantillons
    costs = []  # Liste pour stocker l'évolution du coût
    w = w_random.copy()  # Pour ne pas altérer les poids d'entrée s'il faut les réutiliser

    for i in range(num_iterations):
        # Calcul des prédictions avec les poids actuels
        y_pred = predict(X, w)

        # Calcul du gradient pour les poids (sauf le biais), avec régularisation L1
        dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (lambda_reg / m) * np.sign(w[1:])
        # Calcul du gradient pour le biais (le biais n'est pas régularisé)
        db = (1 / m) * np.sum(y_pred - y)

        # Mise à jour des poids (sauf le biais)
        w[1:] -= learning_rate * dw
        # Mise à jour du biais
        w[0] -= learning_rate * db

        # Calcul et stockage du coût, en incluant le terme de régularisation
        cost = cost_function(X, y, w) + (lambda_reg / m) * np.sum(np.abs(w[1:]))
        costs.append(cost)

    return w, costs





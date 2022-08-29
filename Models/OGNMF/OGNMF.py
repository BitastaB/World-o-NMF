import statistics
import warnings

import numpy as np
import scipy.io
from sklearn.cluster import KMeans

from Utils.metrics_evaluation import evaluate_nmi, accuracy
from Utils.utils import KNN, init_kmeans


def init_params(matS, m, n, k):
    diag = np.sum(matS, axis=1)
    matD = np.diag(diag)
    matL = matD - matS

    H = np.random.rand(m, k)
    V = np.random.rand(k, n)
    return matD, matL, H, V


def updateH(matX, matH, matV, eps_1, eps_2):
    num = matX @ matV.T
    den = matH @ matV @ matV.T
    den[den < eps_1] = eps_2
    res = num / den
    matH = np.multiply(matH, np.sqrt(res))
    return matH


def updateV(matX, matS, matH, matV, alpha, beta, eps_1, eps_2, matD):
    num = (matH.T @ matX) + (alpha * matV @ matS) + (beta * matV)
    den = (matH.T @ matH @ matV) + (alpha * matV @ matD) + (beta * matV @ matV.T @ matV)
    den[den < eps_1] = eps_2
    res = num / den
    matV = np.multiply(matV, np.power(res, 0.25))
    return matV


def GONMF(matX, matS, matH, matV, alpha, beta, eps_1, eps_2, max_iter, matD, n,  kappa=1e-4):
    i = 1
    lst_recon_err = [((np.linalg.norm(matX - (matH @ matV), 'fro'))**2)/n]
    err = 1
    while ((err) >= (kappa)) and (i <= max_iter):
        matH = updateH(matX, matH, matV, eps_1, eps_2)
        matV = updateV(matX, matS, matH, matV, alpha, beta, eps_1, eps_2, matD)

        ## Reconstructed matix
        matX_reconstructed = matH @ matV

        ## Reconstruction Error
        a = np.linalg.norm(matX - matX_reconstructed, 'fro')
        recon_reeor = (a ** 2) / n
        lst_recon_err.append(recon_reeor)

        err = (lst_recon_err[i - 1] - lst_recon_err[i]) / max(1, lst_recon_err[i - 1])

        i += 1
    return matH, matV, recon_reeor


def run_model(matImg, y, alpha_range, beta_range, knn_neigh_list, k_list, maxiter_kmeans, eps_1, eps_2, max_iter):

    # Normalise data
    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0


    ## Definiton of X
    X = normal_img.T
    m, n = X.shape

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    parameters = []
    for a in alpha_range:
        for b in beta_range:
            parameters.append((a, b))

    for k_knn in knn_neigh_list:
        ## Affinity matrix
        matS, matdist, _ = KNN(normal_img, k_knn)

        for k in k_list:
            matD, matL, matH, matV = init_params(matS, m, n, k)

            max_lst_acc_k = []
            max_lst_nmi_k = []
            max_lst_recon_err_k = []

            avg_lst_acc_k = []
            avg_lst_nmi_k = []
            avg_lst_recon_err_k = []

            for p in range(len(parameters)):
                alpha = parameters[p][0]
                beta = parameters[p][1]

                matH, matV, recon_reeor = GONMF(X, matS, matH, matV, alpha, beta, eps_1, eps_2, max_iter, matD, n)

                # Kmeans task
                lst_acc = []
                lst_nmi = []
                lst_recon_err = []

                for i in range(1, maxiter_kmeans):
                    pred = kmeans.fit_predict(matV.T)

                    # NMI
                    nmi = 100 * evaluate_nmi(y, pred)

                    # ACC
                    acc = 100 * accuracy(y, pred)

                    lst_acc.append(acc)
                    lst_nmi.append(nmi)
                    lst_recon_err.append(recon_reeor)
                    # End for

                # Add max values to list for the combo of alpha-beta
                max_lst_acc_k.append(max(lst_acc))
                max_lst_nmi_k.append(max(lst_nmi))
                max_lst_recon_err_k.append(max(lst_recon_err))

                # Add avg values to list for the combo of alpha-beta
                avg_lst_acc_k.append(statistics.mean(lst_acc))
                avg_lst_nmi_k.append(statistics.mean(lst_nmi))
                avg_lst_recon_err_k.append(statistics.mean(lst_recon_err))

            print(
                "**********************************************************************************************************")
            print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs\n")
            print(
                f"k = {k} : best max_acc = {max(max_lst_acc_k)} , with parameters = {parameters[np.argmax(max_lst_acc_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best max_nmi = {max(max_lst_nmi_k)} , with parameters = {parameters[np.argmax(max_lst_nmi_k)]}, k_knn = {k_knn}")
            print("\n\nThe results of running the Kmeans method 20 times and the report of average of 20 runs\n")
            print(
                f"k = {k} : best avg_acc = {max(avg_lst_acc_k)} , with parameters = {parameters[np.argmax(avg_lst_acc_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best avg_nmi = {max(avg_lst_nmi_k)} , with parameters = {parameters[np.argmax(avg_lst_nmi_k)]}, k_knn = {k_knn}")
            print(
                "**********************************************************************************************************")

    print(f"done!")

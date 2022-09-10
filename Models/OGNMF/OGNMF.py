import statistics
import warnings

import numpy as np
import scipy.io
from sklearn.cluster import KMeans

from Utils.metrics_evaluation import evaluate_nmi, accuracy, calculate_silhouette_score, calculate_davies_bouldin_score, \
    calculate_dunn_index
from Utils.utils import KNN, init_kmeans, store_kmeans


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
    return matH, matV, recon_reeor, i - 1


def run_model(model, dataset, matImg, y, alpha_range, beta_range, knn_neigh_list, k_list, maxiter_kmeans, eps_1, eps_2, max_iter):

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

    # For convergence comparison
    iterations = []
    iterations_k2 = {}

    # Util for plotting best cluster produced
    best_cluster_acc = {'acc': 0}

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
            max_lst_sil_score_k = []
            max_lst_dunn_score_k = []
            min_lst_davis_score_k = []

            avg_lst_acc_k = []
            avg_lst_nmi_k = []
            avg_lst_recon_err_k = []
            avg_lst_sil_score_k = []
            avg_lst_dunn_score_k = []
            avg_lst_davis_score_k = []

            for p in range(len(parameters)):
                alpha = parameters[p][0]
                beta = parameters[p][1]

                matH, matV, recon_reeor, n_iteration = GONMF(X, matS, matH, matV, alpha, beta, eps_1, eps_2, max_iter, matD, n)

                iterations.append(n_iteration)

                # Kmeans task
                lst_acc = []
                lst_nmi = []
                lst_recon_err = []
                lst_sil_score = []
                lst_dunn_score = []
                lst_davis_score = []

                for i in range(1, maxiter_kmeans):
                    pred = kmeans.fit_predict(matV.T)

                    # NMI
                    nmi = 100 * evaluate_nmi(y, pred)

                    # ACC
                    acc = 100 * accuracy(y, pred)
                    if acc > best_cluster_acc['acc']:
                        best_cluster_acc['acc'] = acc
                        best_cluster_acc['data'] = matV
                        best_cluster_acc['pred'] = pred


                    # Silhoutte score
                    silhouette_score = calculate_silhouette_score(matV.T, pred)

                    # Davis-bouldin score
                    davis_score = calculate_davies_bouldin_score(matV.T, pred)

                    # dunn's index
                    dunn_score = calculate_dunn_index(matV.T, y)

                    lst_acc.append(acc)
                    lst_nmi.append(nmi)
                    lst_recon_err.append(recon_reeor)
                    lst_sil_score.append(silhouette_score)
                    lst_davis_score.append(davis_score)
                    lst_dunn_score.append(dunn_score)

                    # End for

                # Add max values to list for the combo of alpha-beta
                max_lst_acc_k.append(max(lst_acc))
                max_lst_nmi_k.append(max(lst_nmi))
                max_lst_recon_err_k.append(max(lst_recon_err))
                max_lst_sil_score_k.append(max(lst_sil_score))
                max_lst_dunn_score_k.append((max(lst_dunn_score)))
                min_lst_davis_score_k.append(min(lst_davis_score))

                # Add avg values to list for the combo of alpha-beta
                avg_lst_acc_k.append(statistics.mean(lst_acc))
                avg_lst_nmi_k.append(statistics.mean(lst_nmi))
                avg_lst_recon_err_k.append(statistics.mean(lst_recon_err))
                avg_lst_sil_score_k.append(statistics.mean(lst_sil_score))
                avg_lst_davis_score_k.append(statistics.mean(lst_davis_score))
                avg_lst_dunn_score_k.append(statistics.mean(lst_dunn_score))

            if k not in iterations_k2.keys():
                iterations_k2[k] = [n_iteration]
            else:
                iterations_k2[k].append(n_iteration)

            print(
                "**********************************************************************************************************")
            print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs\n")
            print(
                f"k = {k} : best max acc = {max(max_lst_acc_k)} , with parameters = {parameters[np.argmax(max_lst_acc_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best max nmi = {max(max_lst_nmi_k)} , with parameters = {parameters[np.argmax(max_lst_nmi_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best max Silhoutte score = {max(max_lst_sil_score_k)} , with parameters = {parameters[np.argmax(max_lst_sil_score_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best max Dunn's Index score = {max(max_lst_dunn_score_k)} , with parameters = {parameters[np.argmax(max_lst_dunn_score_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best min Davies Bouldin score = {min(min_lst_davis_score_k)} , with parameters = {parameters[np.argmin(min_lst_davis_score_k)]}, k_knn = {k_knn}")

            print("\n\nThe results of running the Kmeans method 20 times and the report of average of 20 runs\n")
            print(
                f"k = {k} : best avg acc = {max(avg_lst_acc_k)} , with parameters = {parameters[np.argmax(avg_lst_acc_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best avg nmi = {max(avg_lst_nmi_k)} , with parameters = {parameters[np.argmax(avg_lst_nmi_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best avg Silhoutte score = {max(avg_lst_sil_score_k)} , with parameters = {parameters[np.argmax(avg_lst_sil_score_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best avg Dunn's Index score = {max(avg_lst_dunn_score_k)} , with parameters = {parameters[np.argmax(avg_lst_dunn_score_k)]}, k_knn = {k_knn}")
            print(
                f"k = {k} : best avg Davies Bouldin score = {min(avg_lst_davis_score_k)} , with parameters = {parameters[np.argmin(avg_lst_davis_score_k)]}, k_knn = {k_knn}")

            print(
                "**********************************************************************************************************")

    ## print for convergence comparison
    for k in k_list:
        print(f"Average no. of iterations for k = {k} : {statistics.mean(iterations_k2[k])}")
    print(f"Overall average no. of iterations : {statistics.mean(iterations)}")

    # Storing details of best cluster
    data = best_cluster_acc['data']
    pred = best_cluster_acc['pred']
    store_kmeans(data, pred, model, dataset)

    print(f"done!")

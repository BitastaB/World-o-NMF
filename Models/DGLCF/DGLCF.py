import statistics

import numpy as np

from Utils.metrics_evaluation import accuracy, evaluate_nmi, calculate_silhouette_score, calculate_davies_bouldin_score, \
    calculate_dunn_index
from Utils.utils import init_kmeans, KNN, store_kmeans


def DGLCF(X, e, DV, DU, WV, WU, beta, _lambda, nu, k, n, maxiter):
    U = np.random.rand(n, k)
    V = np.random.rand(n, k)
    iter = 1
    E = {iter: ((np.linalg.norm(X - (X @ U @ V.T), 'fro')) ** 2) / n}
    err = 1
    while (iter <= maxiter and err >= 1e-10):
        ##Update U
        numU = (X.T @ X @ V) + (beta * WU @ U)
        denU = (X.T @ X @ U @ V.T @ V) + (beta * DU @ U)
        denU[denU < 1e-10] = 1e-10
        re1 = np.divide(numU, denU)
        U = np.multiply(U, re1)

        ## Update V
        numV = (X.T @ X @ U) + (_lambda * WV @ V) + (nu * V)
        denV = (V @ U.T @ X.T @ X @ U) + (_lambda * DV @ V) + (nu * e * V)
        denV[denV < 1e-10] = 1e-10
        re2 = np.divide(numV, denV)
        V = np.multiply(V, re2)

        iter += 1
        E[iter] = ((np.linalg.norm(X - (X @ U @ V.T), 'fro') ** 2) / n)
        err = (E[iter - 1] - E[iter]) / max(1, E[iter - 1])

    return U, V, iter - 1


def run_model(model, dataset, matImg, y, beta_range, lambda_range, knn_neigh_list, k_list, maxiter, maxiter_kmeans, nu_range=[0.1, 10]):
    # Normalise data
    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0

    # Definition of X
    X = normal_img.T
    m, n = (X.T @ X).shape

    onesn = np.ones(n)
    e = (1 / n) * onesn @ onesn.T

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    # For convergence comparison
    iterations = []
    iterations_k2 = {}

    # Util for plotting best cluster produced
    best_cluster_acc = {'acc': 0}

    parameters = []
    for beta in beta_range:
        for _lambda in lambda_range:
            for nu in nu_range:
                parameters.append((beta, _lambda, nu))

    for k_knn in knn_neigh_list:

        ## Affinity matrix
        WV, DV, _ = KNN(X.T, k_knn)
        WS, DS, _ = KNN(X, k_knn)
        DU = X.T @ DS @ X
        WU = X.T @ WS @ X

        for k in k_list:

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
                beta = parameters[p][0]
                _lambda = parameters[p][1]
                nu = parameters[p][2]

                U, V, n_iteration = DGLCF(X, e, DV, DU, WV, WU, beta, _lambda, nu, k, n, maxiter)

                X_recon = U @ V.T
                iterations.append(n_iteration)

                # Kmeans task
                lst_acc = []
                lst_nmi = []
                lst_recon_err = []
                lst_sil_score = []
                lst_dunn_score = []
                lst_davis_score = []

                for i in range(1, maxiter_kmeans):
                    pred = kmeans.fit_predict(V)

                    # NMI
                    nmi = 100 * evaluate_nmi(y, pred)

                    # ACC
                    acc = 100 * accuracy(y, pred)
                    if acc > best_cluster_acc['acc']:
                        best_cluster_acc['acc'] = acc
                        best_cluster_acc['recon'] = X_recon
                        best_cluster_acc['pred'] = pred
                        best_cluster_acc['data'] = V.T

                    # Silhoutte score
                    silhouette_score = calculate_silhouette_score(V, pred)

                    # Davis-bouldin score
                    davis_score = calculate_davies_bouldin_score(V, pred)

                    # dunn's index
                    dunn_score = calculate_dunn_index(V, y)


                    lst_acc.append(acc)
                    lst_nmi.append(nmi)
#                    lst_recon_err.append(recon_reeor)
                    lst_sil_score.append(silhouette_score)
                    lst_davis_score.append(davis_score)
                    lst_dunn_score.append(dunn_score)

                    # End for

                # Add max values to list for the combo of alpha-beta
                max_lst_acc_k.append(max(lst_acc))
                max_lst_nmi_k.append(max(lst_nmi))
 #               max_lst_recon_err_k.append(max(lst_recon_err))
                max_lst_sil_score_k.append(max(lst_sil_score))
                max_lst_dunn_score_k.append((max(lst_dunn_score)))
                min_lst_davis_score_k.append(min(lst_davis_score))

                # Add avg values to list for the combo of alpha-beta
                avg_lst_acc_k.append(statistics.mean(lst_acc))
                avg_lst_nmi_k.append(statistics.mean(lst_nmi))
#                avg_lst_recon_err_k.append(statistics.mean(lst_recon_err))
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
    data_recon = best_cluster_acc['recon']
   # store_kmeans(data, pred, data_recon, model, dataset)

    print(f"done!")


import statistics

import numpy as np

from Utils.metrics_evaluation import evaluate_nmi, accuracy, calculate_silhouette_score
from Utils.utils import init_kmeans


def computeW(X, S, H, nu):
    Z = np.square(X - (S @ H))
    eps = 2 ** (-52)
    e = np.exp(1)
    C = np.power(e, -np.sum(Z, axis=1) / nu) + eps
    D = np.sum(C, axis=0)
    W = np.diag(C / D)
    return W


def ERWNMF(X, nu, k, m, n, maxiter):

    # Initialisation
    S = np.random.rand(m, k)
    H = np.random.rand(k, n)
    W = computeW(X, S, H, nu)
    iter = 1
    while iter <= maxiter:
        # Update W
        W = computeW(X, S, H, nu)

        # Update S
        numS = X @ H.T
        denS = S @ H @ H.T
        denS[denS < 1e-10] = 1e-10
        re1 = np.divide(numS, denS)
        S = np.multiply(S, re1)

        ##Update H
        WS = W @ S
        numH = WS.T @ X
        denH = WS.T @ S @ H
        denH[denH < 1e-10] = 1e-10
        re2 = np.divide(numH, denH)
        H = np.multiply(H, re2)
        iter += 1

    return W, S, H


def run_model(matImg, y, k_list, maxiter, maxiter_kmeans, plot_graphs):

    best_nmi = 0
    best_acc = 0
    best_sil_score = 0
    best_k_acc = 0
    best_k_nmi = 0
    best_k_sil_Score = 0

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

    a = list(range(22, 33))
    nu_list = np.power(2, a)

    for k in k_list:
        max_lst_acc_k = []
        max_lst_nmi_k = []
        max_lst_sil_score_k = []

        avg_lst_acc_k = []
        avg_lst_nmi_k = []
        avg_lst_sil_score_k = []

        for nu in nu_list:

            W, S, H = ERWNMF(X, nu, k, m, n, maxiter)

            ## Kmeans task
            lst_acc = []
            lst_nmi = []
            lst_sil_score = []

            for i in range(1, maxiter_kmeans):
                pred = kmeans.fit_predict(H.T)

                # NMI
                nmi = 100 * evaluate_nmi(y, pred)

                # ACC
                acc = 100 * accuracy(y, pred)

                #Silhoutte score
                silhouette_score = calculate_silhouette_score(H.T, pred)

                lst_acc.append(acc)
                lst_nmi.append(nmi)
                lst_sil_score.append(silhouette_score)
                # End for

            # Add max values to list for one theta
            max_lst_acc_k.append(max(lst_acc))
            max_lst_nmi_k.append(max(lst_nmi))
            max_lst_sil_score_k.append(max(lst_sil_score))

            # Add avg values to list for one theta
            avg_lst_acc_k.append(statistics.mean(lst_acc))
            avg_lst_nmi_k.append(statistics.mean(lst_nmi))
            avg_lst_sil_score_k.append(statistics.mean(lst_sil_score))

        if max(avg_lst_acc_k) > best_acc:
            best_acc = max(avg_lst_acc_k)
            best_k_acc = k
        if max(avg_lst_nmi_k) > best_nmi:
            best_nmi = max(avg_lst_nmi_k)
            best_k_nmi = k
        if max(avg_lst_sil_score_k) > best_sil_score:
            best_sil_score = max(avg_lst_sil_score_k)
            best_k_sil_Score = k


        print(
            "**********************************************************************************************************")
        print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs\n")
        print(f"k = {k} : best max_acc = {max(max_lst_acc_k)} , with nu = {nu_list[np.argmax(max_lst_acc_k)]}")
        print(f"k = {k} : best max_nmi = {max(max_lst_nmi_k)} , with nu = {nu_list[np.argmax(max_lst_nmi_k)]}")
        print(f"k = {k} : best max_silhoutte score = {max(max_lst_sil_score_k)} , with nu = {nu_list[np.argmax(max_lst_sil_score_k)]}")

        print("\n\nThe results of running the Kmeans method 20 times and the report of average of 20 runs\n")
        print(f"k = {k} : best avg_acc = {max(avg_lst_acc_k)} , with nu = {nu_list[np.argmax(avg_lst_acc_k)]} ")
        print(f"k = {k} : best avg_nmi = {max(avg_lst_nmi_k)} , with nu = {nu_list[np.argmax(avg_lst_nmi_k)]} ")
        print(f"k = {k} : best avg_silhoutte score = {max(avg_lst_sil_score_k)} , with nu = {nu_list[np.argmax(avg_lst_sil_score_k)]} ")
        print(
            "**********************************************************************************************************")

    print("\n....................................................")
    print(f" best acc(avg) = {best_acc} for k = {best_k_acc}")
    print(f" best nmi(avg) = {best_nmi} for k = {best_k_nmi}")
    print("....................................................")

    print(f"done!")

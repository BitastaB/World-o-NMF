import numpy as np
from Utils.metrics_evaluation import evaluate_nmi, accuracy, calculate_silhouette_score, calculate_davies_bouldin_score, \
    calculate_dunn_index
import scipy.io
from numpy.linalg import linalg
from sklearn.cluster import KMeans
import statistics
import warnings

from Utils.utils import KNN, init_kmeans

warnings.filterwarnings('ignore')


def L21(Z):
    z = np.diag(Z @ Z.T)
    z.setflags(write=True)
    z[z < 1e-10] = 1e-10
    z = 4 * z
    D = np.diag(np.sqrt(np.divide(1, z)))
    return D


def pospart(A):
    B = 0.5 * (abs(A) + A)
    return B


def negpart(A):
    B = 0.5 * (abs(A) - A)
    return B


def RSCNMF(X, XX, e, S, D, alpha, beta, _lambda, l, n, maxiter):
    W = np.random.rand(n, l)
    H = np.random.rand(l, n)

    Z = X - (X @ W @ H)
    Q = L21(Z.T)
    P = L21(W.T)
    t = 1

    E = {t: np.trace(Z @ Q @ Z.T) + (alpha * np.trace(H @ (D - S) @ H.T)) + (
            beta * np.trace(H @ (np.identity(n) - e) @ H.T)) + (_lambda * np.trace(W @ P @ W.T))}

    err = 1

    while t <= maxiter and err >= 1e-06:
        # Update W
        numW = (pospart(XX) @ Q @ H.T) + (negpart(XX) @ W @ H @ Q @ H.T)
        denW = (negpart(XX) @ Q @ H.T) + (pospart(XX) @ W @ H @ Q @ H.T) + (_lambda * W @ P)
        denW[denW < 1e-10] = 1e-10
        re1 = np.divide(numW, denW)
        re1 = np.sqrt(re1)
        W = np.multiply(W, re1)

        # Update H
        numH = (W.T @ pospart(XX) @ Q) + (W.T @ negpart(XX) @ W @ H @ Q) + (alpha * H @ S) + (beta * H)
        denH = (W.T @ negpart(XX) @ Q) + (W.T @ pospart(XX) @ W @ H @ Q) + (alpha * H @ D) + (beta * H * e)
        denH[denH < 1e-10] = 1e-10
        re2 = np.divide(numH, denH)
        re2 = np.sqrt(re2)
        H = np.multiply(H, re2)

        # Update Z
        Z = X - (X @ W @ H)

        # Update Q
        Q = L21(Z.T)

        # Update P
        P = L21(W.T)

        t += 1
        E[t] = np.trace(Z @ Q @ Z.T) + (alpha * np.trace(H @ (D - S) @ H.T)) + (
                beta * np.trace(H @ (np.identity(n) - e) @ H.T)) + (_lambda * np.trace(W @ P @ W.T))

        err = abs(E[t - 1] - E[t]) / abs(E[t - 1])

    return W, H, t - 1


def run_model(matImg, y, alpha_list, beta_list, k_knn_range, k_list, lambda_list, maxiter_kmeans, maxiter):

    # Normalise data
    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0

    # Definiton of X
    X = normal_img.T
    m, n = X.shape
    XX = X.T @ X

    # Setting params and hyper params
    onesn = np.ones(n)
    E = (1 / n) * onesn @ onesn.T

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    # For convergence comparison
    iterations = []
    iterations_k2 = {}

    parameters = []
    for a in alpha_list:
        for b in beta_list:
            parameters.append((a, b))

    for knn in k_knn_range:
        S, _, _ = KNN(normal_img, knn)
        diag = np.sum(S, axis=1)
        D = np.diag(diag)
        L = D - S

        for k in k_list:

            maxAcc = {}
            maxNmi = {}
            maxSilScore = {}
            maxDunnScore = {}
            minDavisScore = {}

            ##
            meanAcc = {}
            meanNmi = {}
            meanSilScore = {}
            meanDunnScore = {}
            meanDavisScore = {}

            for p in parameters:

                maxlst_acc = []
                maxlst_nmi = []
                maxlst_sil_score = []
                maxlst_dunn_score = []
                minlst_davis_score = []

                ##
                meanlst_acc = []
                meanlst_nmi = []
                meanlst_sil_score = []
                meanlst_dunn_score = []
                meanlst_davis_score = []

                for _lambda in lambda_list:
                    W, H, n_iteration = RSCNMF(X, XX, E, S, D, p[0], p[1], _lambda, k, n, maxiter)

                    iterations.append(n_iteration)

                    ## Kmeans task
                    lst_acc = []
                    lst_nmi = []
                    lst_sil_score = []
                    lst_dunn_score = []
                    lst_davis_score = []

                    for i in range(1, maxiter_kmeans):
                        pred = []
                        pred = kmeans.fit_predict(H.T)

                        # NMI
                        nmi = 100 * evaluate_nmi(y, pred)

                        # ACC
                        acc = 100 * accuracy(y, pred)

                        # Silhoutte score
                        silhouette_score = calculate_silhouette_score(H.T, pred)

                        # Davis-bouldin score
                        davis_score = calculate_davies_bouldin_score(H.T, pred)

                        # dunn's index
                        dunn_score = calculate_dunn_index(H.T, y)

                        lst_acc.append(acc)
                        lst_nmi.append(nmi)
                        lst_sil_score.append(silhouette_score)
                        lst_davis_score.append(davis_score)
                        lst_dunn_score.append(dunn_score)

                        ## End for

                        ##
                    maxlst_acc.append(max(lst_acc))
                    maxlst_nmi.append(max(lst_nmi))
                    maxlst_sil_score.append(max(lst_sil_score))
                    maxlst_dunn_score.append((max(lst_dunn_score)))
                    minlst_davis_score.append(min(lst_davis_score))

                    ##
                    meanlst_acc.append(statistics.mean(lst_acc))
                    meanlst_nmi.append(statistics.mean(lst_nmi))
                    meanlst_sil_score.append(statistics.mean(lst_sil_score))
                    meanlst_davis_score.append(statistics.mean(lst_davis_score))
                    meanlst_dunn_score.append(statistics.mean(lst_dunn_score))

                    ## End for

                maxAcc[p] = maxlst_acc
                maxNmi[p] = maxlst_nmi
                maxSilScore[p] = maxlst_sil_score
                maxDunnScore[p] = maxlst_dunn_score
                minDavisScore[p] = minlst_davis_score

                ##
                meanAcc[p] = meanlst_acc
                meanNmi[p] = meanlst_nmi
                meanSilScore[p] = meanlst_sil_score
                meanDavisScore[p] = meanlst_davis_score
                meanDunnScore[p] = meanlst_dunn_score


            if k not in iterations_k2.keys():
                iterations_k2[k] = [n_iteration]
            else:
                iterations_k2[k] = iterations_k2[k].append(n_iteration)


                ## ENd for k2

            maxacc_final = {}
            maxnmi_final = {}
            maxSilScore_final = {}
            maxDunnScore_final = {}
            minDavisScore_final = {}

            print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs")
            for p in parameters:
                maxacc_final[p] = [max(maxAcc[p]), lambda_list[np.argmax(maxAcc[p])]]
                maxnmi_final[p] = [max(maxNmi[p]), lambda_list[np.argmax(maxNmi[p])]]
                maxSilScore_final[p] = [max(maxSilScore[p]), lambda_list[np.argmax(maxSilScore[p])]]
                maxDunnScore_final[p] = [max(maxDunnScore[p]), lambda_list[np.argmax(maxDunnScore[p])]]
                minDavisScore_final[p] = [min(minDavisScore[p]), lambda_list[np.argmin(minDavisScore[p])]]

                print(
                    f"##################################################################################################")
                print(f" k = {k} : alpha,beta = {p} : k_knn = {knn}")
                print(f" Max ACC : {maxacc_final[p][0]}, with lambda = {lambda_list[np.argmax(maxAcc[p])]}")
                print(f" Max NMI : {maxnmi_final[p][0]}, with lambda = {lambda_list[np.argmax(maxNmi[p])]}")
                print(f" Max Silhoutte score : {maxSilScore_final[p][0]}, with lambda = {lambda_list[np.argmax(maxSilScore[p])]}")
                print(f" Max Dunn's Index score : {maxDunnScore_final[p][0]}, with lambda = {lambda_list[np.argmax(maxDunnScore[p])]}")
                print(f" Min Davies Bouldin score : {minDavisScore_final[p][0]}, with lambda = {lambda_list[np.argmin(minDavisScore[p])]}")

                print(
                    f"##################################################################################################")
            ##

            meanacc_final = {}
            meannmi_final = {}
            meanSilScore_final = {}
            meanDunnScore_final = {}
            meanDavidScore_final = {}

            print("The results of running the Kmeans method 20 times and the average of 20 runs")
            for p in parameters:
                meanacc_final[p] = [max(meanAcc[p]), lambda_list[np.argmax(meanAcc[p])]]
                meannmi_final[p] = [max(meanNmi[p]), lambda_list[np.argmax(meanNmi[p])]]
                meanSilScore_final[p] = [max(meanSilScore[p]), lambda_list[np.argmax(meanSilScore[p])]]
                meanDunnScore_final[p] = [max(meanDunnScore[p]), lambda_list[np.argmax(meanDunnScore[p])]]
                meanDavidScore_final[p] = [min(meanDavisScore[p]), lambda_list[np.argmin(meanDavisScore[p])]]

                print(
                    f"##################################################################################################")
                print(f" k1 = {k} : alpha, beta = {p} : k_knn = {knn}")
                print(f" Avg ACC : {meanacc_final[p][0]}, with lambda = {lambda_list[np.argmax(meanAcc[p])]}")
                print(f" Avg NMI : {meannmi_final[p][0]}, with lambda = {lambda_list[np.argmax(meanNmi[p])]}")
                print(f" Avg Silhoutte score : {meanSilScore_final[p][0]}, with lambda = {lambda_list[np.argmax(meanSilScore[p])]}")
                print(f" Avg Dunn's Index score : {meanDunnScore_final[p][0]}, with lambda = {lambda_list[np.argmax(meanDunnScore[p])]}")
                print(f" Avg Davies Bouldin score : {meanDavidScore_final[p][0]}, with lambda = {lambda_list[np.argmax(meanDavisScore[p])]}")

                print(
                    f"##################################################################################################")


        ##**

        # print for convergence comparison
        print(iterations_k2)
        for k in k_list:
            print(f"Average no. of iterations for k = {k} : {statistics.mean(iterations_k2[k])}")
        print(f"Overall average no. of iterations : {statistics.mean(iterations)}")

        print("Done")

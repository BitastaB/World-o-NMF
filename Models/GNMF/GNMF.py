import numpy as np
from Utils.metrics_evaluation import evaluate_nmi, accuracy, calculate_silhouette_score, calculate_davies_bouldin_score, \
    calculate_dunn_index
import scipy.io
from numpy.linalg import linalg
from sklearn.cluster import KMeans
import statistics
import warnings

from Utils.utils import KNN, init_kmeans, store_kmeans

warnings.filterwarnings('ignore')


def GNMF(X, W, D, _lambda, k, m, n, maxiter):
    U = np.random.rand(m, k)
    V = np.random.rand(n, k)
    iter = 1
    E = {iter: ((np.linalg.norm(X - (U @ V.T), 'fro')) ** 2) / n}
    err = 1

    while iter <= maxiter and err >= 1e-06:
        # Update U
        numU = X @ V
        denU = U @ V.T @ V
        denU[denU < 1e-10] = 1e-10
        re1 = np.divide(numU, denU)
        U = np.multiply(U, re1)

        # Update V
        numV = X.T @ U + _lambda * W @ V
        denV = V @ U.T @ U + _lambda * D @ V
        denV[denV < 1e-10] = 1e-10
        re2 = np.divide(numV, denV)
        V = np.multiply(V, re2)

        iter += 1
        E[iter] = ((np.linalg.norm(X - (U @ V.T), 'fro')) ** 2) / n
        err = (E[iter - 1] - E[iter]) / max(1, E[iter - 1])

    return U, V, iter - 1


def run_model(model, dataset, matImg, y, k_knn_list, k_list, lambda_list, maxiter, maxiter_kmeans):

    # Normalise data
    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0


    # Definiton of X
    X = normal_img.T
    m, n = X.shape

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    # Util for plotting best cluster produced
    best_cluster_acc = {'acc': 0}

    # For convergence comparison
    iterations = []
    iterations_k2 = {}

    for knn in k_knn_list:
        W, _, _ = KNN(normal_img, knn)
        diag = np.sum(W, axis=1)
        D = np.diag(diag)
        L = D - W

        maxAcc = {}
        maxNmi = {}
        maxRecon_reeor = {}
        maxSilScore = {}
        maxDunnScore = {}
        minDavisScore = {}

        ##
        meanAcc = {}
        meanNmi = {}
        meanRecon_reeor = {}
        meanSilScore = {}
        meanDunnScore = {}
        meanDavisScore = {}

        for k in k_list:

            maxlst_acc = []
            maxlst_nmi = []
            maxlst_recon_err = []
            maxlst_sil_score = []
            maxlst_dunn_score = []
            minlst_davis_score = []

            ##
            meanlst_acc = []
            meanlst_nmi = []
            meanlst_recon_err = []
            meanlst_sil_score = []
            meanlst_dunn_score = []
            meanlst_davis_score = []

            for _lambda in lambda_list:
                U, V, n_iteration = GNMF(X, W, D, _lambda, k, m, n, maxiter)

                iterations.append(n_iteration)

                ## Reconstructed matix
                X_reconstructed = U @ V.T

                ## Kmeans task
                lst_acc = []
                lst_nmi = []
                lst_recon_err = []
                lst_sil_score = []
                lst_dunn_score = []
                lst_davis_score = []

                for i in range(1, maxiter_kmeans):

                    pred = kmeans.fit_predict(V)

                    ## NMI
                    nmi = 100 * evaluate_nmi(y, pred)

                    ## ACC
                    acc = 100 * accuracy(y, pred)
                    if acc > best_cluster_acc['acc']:
                        best_cluster_acc['acc'] = acc
                        best_cluster_acc['data'] = V.T
                        best_cluster_acc['pred'] = pred


                    # Silhoutte score
                    silhouette_score = calculate_silhouette_score(V, pred)

                    # Davis-bouldin score
                    davis_score = calculate_davies_bouldin_score(V, pred)

                    # dunn's index
                    dunn_score = calculate_dunn_index(V, y)

                    ## Reconstruction Error
                    a = np.linalg.norm(X - X_reconstructed, 'fro')
                    recon_reeor = (a)  # /n

                    lst_acc.append(acc)
                    lst_nmi.append(nmi)
                    lst_recon_err.append(recon_reeor)
                    lst_sil_score.append(silhouette_score)
                    lst_davis_score.append(davis_score)
                    lst_dunn_score.append(dunn_score)

                    ## End for

                ##
                maxlst_acc.append(max(lst_acc))
                maxlst_nmi.append(max(lst_nmi))
                maxlst_recon_err.append(max(lst_recon_err))
                maxlst_sil_score.append(max(lst_sil_score))
                maxlst_dunn_score.append((max(lst_dunn_score)))
                minlst_davis_score.append(min(lst_davis_score))

                ##
                meanlst_acc.append(statistics.mean(lst_acc))
                meanlst_nmi.append(statistics.mean(lst_nmi))
                meanlst_recon_err.append(statistics.mean(lst_recon_err))
                meanlst_sil_score.append(statistics.mean(lst_sil_score))
                meanlst_davis_score.append(statistics.mean(lst_davis_score))
                meanlst_dunn_score.append(statistics.mean(lst_dunn_score))

                if k not in iterations_k2.keys():
                    iterations_k2[k] = [n_iteration]
                else:
                    iterations_k2[k] = iterations_k2[k].append(n_iteration)

                ## End for

            maxAcc[k] = maxlst_acc
            maxNmi[k] = maxlst_nmi
            maxRecon_reeor[k] = maxlst_recon_err
            maxSilScore[k] = maxlst_sil_score
            maxDunnScore[k] = maxlst_dunn_score
            minDavisScore[k] = minlst_davis_score

            ##
            meanAcc[k] = meanlst_acc
            meanNmi[k] = meanlst_nmi
            meanRecon_reeor[k] = meanlst_recon_err
            meanSilScore[k] = meanlst_sil_score
            meanDavisScore[k] = meanlst_davis_score
            meanDunnScore[k] = meanlst_dunn_score



            ## ENd for k2

        maxacc_final = {}
        maxnmi_final = {}
        maxrecon_final = {}
        maxSilScore_final = {}
        maxDunnScore_final = {}
        minDavisScore_final = {}

        print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs")
        for k in k_list:
            maxacc_final[k] = [max(maxAcc[k]), lambda_list[np.argmax(maxAcc[k])]]
            maxnmi_final[k] = [max(maxNmi[k]), lambda_list[np.argmax(maxNmi[k])]]
            maxrecon_final[k] = [max(maxRecon_reeor[k]), lambda_list[np.argmax(maxRecon_reeor[k])]]
            maxSilScore_final[k] = [max(maxSilScore[k]), lambda_list[np.argmax(maxSilScore[k])]]
            maxDunnScore_final[k] = [max(maxDunnScore[k]), lambda_list[np.argmax(maxDunnScore[k])]]
            minDavisScore_final[k] = [min(minDavisScore[k]), lambda_list[np.argmin(minDavisScore[k])]]

            print(f"##################################################################################################")
            print(f" k_knn = {knn}  ")
            print(f" k = {k} :  Max ACC : {maxacc_final[k][0]}, with theta = {lambda_list[np.argmax(maxAcc[k])]}")
            print(f" k = {k} :  Max NMI : {maxnmi_final[k][0]}, with theta = {lambda_list[np.argmax(maxNmi[k])]}")
            print(f" k = {k} :  Reconstruction Error : {maxrecon_final[k][0]}")
            print(f" k = {k} :  Max Silhoutter score : {maxSilScore_final[k][0]}, with theta = "
                  f"{lambda_list[np.argmax(maxSilScore[k])]}")
            print(f" k = {k} :  Max Dunn's Index score : {maxDunnScore_final[k][0]}, with theta = "
                  f"{lambda_list[np.argmax(maxDunnScore[k])]}")
            print(f" k = {k} :  Min David Bouldin score : {minDavisScore[k][0]}, with theta = "
                  f"{lambda_list[np.argmin(minDavisScore[k])]}")

            print(f"##################################################################################################")
        ##

        meanacc_final = {}
        meannmi_final = {}
        meanrecon_final = {}
        meanSilScore_final = {}
        meanDunnScore_final = {}
        meanDavidScore_final = {}

        print("The results of running the Kmeans method 20 times and the average of 20 runs")
        for k in k_list:
            meanacc_final[k] = [max(meanAcc[k]), lambda_list[np.argmax(meanAcc[k])]]
            meannmi_final[k] = [max(meanNmi[k]), lambda_list[np.argmax(meanNmi[k])]]
            meanrecon_final[k] = [max(meanRecon_reeor[k]), lambda_list[np.argmax(meanRecon_reeor[k])]]
            meanSilScore_final[k] = [max(meanSilScore[k]), lambda_list[np.argmax(meanSilScore[k])]]
            meanDunnScore_final[k] = [max(meanDunnScore[k]), lambda_list[np.argmax(meanDunnScore[k])]]
            meanDavidScore_final[k] = [max(meanDavisScore[k]), lambda_list[np.argmin(meanDavisScore[k])]]

            print(f"##################################################################################################")
            print(f" k_knn = {knn} ")
            print(f" k = {k} :  Avg ACC : {meanacc_final[k][0]}, with theta = {lambda_list[np.argmax(meanAcc[k])]}")
            print(f" k = {k} :  Avg NMI : {meannmi_final[k][0]}, with theta = {lambda_list[np.argmax(meanNmi[k])]}")
            print(f" k = {k} :  Reconstruction Error : {meanrecon_final[k][0]}")
            print(f" k = {k} :  Avg Silhoutte score : {meanSilScore_final[k][0]}, with theta = "
                  f"{lambda_list[np.argmax(meanSilScore[k])]}")
            print(f" k = {k} :  Avg Dunn's Index score : {meanDunnScore_final[k][0]}, with theta = "
                  f"{lambda_list[np.argmax(meanDunnScore[k])]}")
            print(f" k = {k} :  Avg David Bouldin score : {meanDavidScore_final[k][0]}, with theta = "
                  f"{lambda_list[np.argmin(meanDavisScore[k])]}")

            print(f"##################################################################################################")
    ##**
    ## print for convergence comparison
    print(iterations_k2)
    for k in k_list:
        print(f"Average no. of iterations for k = {k} : {statistics.mean(iterations_k2[k])}")
    print(f"Overall average no. of iterations : {statistics.mean(iterations)}")

    # Storing details of best cluster
    data = best_cluster_acc['data']
    pred = best_cluster_acc['pred']
    store_kmeans(data, pred, model, dataset)

    print("done")

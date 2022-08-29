import numpy as np
from Utils.metrics_evaluation import evaluate_nmi, accuracy
import scipy.io
from numpy.linalg import linalg
from sklearn.cluster import KMeans
import statistics
import warnings

from Utils.utils import KNN, init_kmeans

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

    return U, V


def run_model(matImg, y, k_knn_list, k_list, lambda_list, maxiter, maxiter_kmeans):

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

    for knn in k_knn_list:
        W, _, _ = KNN(normal_img, knn)
        diag = np.sum(W, axis=1)
        D = np.diag(diag)
        L = D - W

        maxAcc = {}
        maxNmi = {}
        maxRecon_reeor = {}
        ##
        meanAcc = {}
        meanNmi = {}
        meanRecon_reeor = {}

        for k in k_list:

            maxlst_acc = []
            maxlst_nmi = []
            maxlst_recon_err = []
            ##
            meanlst_acc = []
            meanlst_nmi = []
            meanlst_recon_err = []

            for _lambda in lambda_list:
                U, V = GNMF(X, W, D, _lambda, k, m, n, maxiter)

                ## Reconstructed matix
                X_reconstructed = U @ V.T

                ## Kmeans task
                lst_acc = []
                lst_nmi = []
                lst_recon_err = []

                for i in range(1, maxiter_kmeans):
                    pred = []
                    pred = kmeans.fit_predict(V)

                    ## NMI
                    nmi = 100 * evaluate_nmi(y, pred)

                    ## ACC
                    acc = 100 * accuracy(y, pred)

                    ## Reconstruction Error
                    a = np.linalg.norm(X - X_reconstructed, 'fro')
                    recon_reeor = (a)  # /n

                    lst_acc.append(acc)
                    lst_nmi.append(nmi)
                    lst_recon_err.append(recon_reeor)
                    ## End for

                ##
                maxlst_acc.append(max(lst_acc))
                maxlst_nmi.append(max(lst_nmi))
                maxlst_recon_err.append(max(lst_recon_err))

                ##
                meanlst_acc.append(statistics.mean(lst_acc))
                meanlst_nmi.append(statistics.mean(lst_nmi))
                meanlst_recon_err.append(statistics.mean(lst_recon_err))
                ## End for

            maxAcc[k] = maxlst_acc
            maxNmi[k] = maxlst_nmi
            maxRecon_reeor[k] = maxlst_recon_err

            ##
            meanAcc[k] = meanlst_acc
            meanNmi[k] = meanlst_nmi
            meanRecon_reeor[k] = meanlst_recon_err
            ## ENd for k2

        maxacc_final = {}
        maxnmi_final = {}
        maxrecon_final = {}

        print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs")
        for k in k_list:
            maxacc_final[k] = [max(maxAcc[k]), lambda_list[np.argmax(maxAcc[k])]]
            maxnmi_final[k] = [max(maxNmi[k]), lambda_list[np.argmax(maxNmi[k])]]
            maxrecon_final[k] = [max(maxRecon_reeor[k]), lambda_list[np.argmax(maxRecon_reeor[k])]]
            print(f"##################################################################################################")
            print(f" k = {k} :  k_knn = {knn}  ")
            print(f" ACC : {maxacc_final[k][0]}, with theta = {lambda_list[np.argmax(maxAcc[k])]}")
            print(f" NMI : {maxnmi_final[k][0]}, with theta = {lambda_list[np.argmax(maxNmi[k])]}")
            print(f" Reconstruction Error : {maxrecon_final[k][0]}")
            print(f"##################################################################################################")
        ##

        meanacc_final = {}
        meannmi_final = {}
        meanrecon_final = {}

        print("The results of running the Kmeans method 20 times and the average of 20 runs")
        for k in k_list:
            meanacc_final[k] = [max(meanAcc[k]), lambda_list[np.argmax(meanAcc[k])]]
            meannmi_final[k] = [max(meanNmi[k]), lambda_list[np.argmax(meanNmi[k])]]
            meanrecon_final[k] = [max(meanRecon_reeor[k]), lambda_list[np.argmax(meanRecon_reeor[k])]]
            print(f"##################################################################################################")
            print(f"k = {k} :  k_knn = {knn} ")
            print(f" ACC : {meanacc_final[k][0]}, with lambda = {lambda_list[np.argmax(meanAcc[k])]}")
            print(f" NMI : {meannmi_final[k][0]}, with lambda = {lambda_list[np.argmax(meanNmi[k])]}")
            print(f" Reconstruction Error : {meanrecon_final[k][0]}")
            print(f"##################################################################################################")
    ##**
    print("done")

""" Based on paper dnsNMF(2018) """

from Utils.metrics_evaluation import *
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import warnings
import statistics

from Utils.utils import init_kmeans

warnings.filterwarnings('ignore')


def constructS(theta, q):
    A = (1 - theta) * np.eye(q)
    B = (theta / q) * np.ones((q, q))
    S = A + B
    return S


def constructB(Z, H, thetha, r, m, start, final):
    P = constructS(thetha, r[start - 1])
    for i in range(start, final + 1):
        S = constructS(thetha, r[i])
        P = P @ Z[i] @ S
    B = P @ H[m]
    return B


def obj(X, Z, H, thetha, r, l, p, n):
    P = np.identity(p)
    for i in range(1, l + 1):
        S = constructS(thetha, r[i])
        P = P @ Z[i] @ S

    f = ((np.linalg.norm(X - (P @ H[l]), 'fro')) ** 2) / n
    ff = 0.5 * ((np.linalg.norm(X - (P @ H[l]), 'fro')) ** 2)
    return f, ff


def pretrain(X, l, r):
    W = X
    Z = {}
    H = {}
    for i in range(1, l + 1):
        r_i = r[i]
        nmf_model = NMF(r_i, init='nndsvd')
        nmf_features = nmf_model.fit_transform(W)
        Z[i] = nmf_features
        nmf_components = nmf_model.components_
        H[i] = nmf_components
        W = H[i]
    return Z, H


def computeZ1(Q, B, p, n, r, maxiter_inner):
    X = Q.T
    A = B.T
    iter = 1
    H = {iter: np.random.rand(r, p)}
    Y = {iter: H[iter]}
    AA = A.T @ A

    alpha = {iter: 1}
    E = {iter: ((np.linalg.norm(X - (A @ H[iter]), 'fro')) ** 2) / n}

    L = np.linalg.norm(AA)
    err = 1

    while iter <= maxiter_inner and err >= 1e-06:
        iter += 1
        GradHY = (AA @ Y[iter - 1]) - (A.T @ X)
        H[iter] = Y[iter - 1] - GradHY / L
        H[iter][H[iter] < 0] = 0
        alpha[iter] = (1 + np.sqrt((4 * alpha[iter - 1]) ** 2) + 1) / 2
        Y[iter] = H[iter] + ((alpha[iter - 1] / alpha[iter]) * (H[iter] - H[iter - 1]))
        E[iter] = ((np.linalg.norm(X - (A @ H[iter]), 'fro')) ** 2) / n
        err = (E[iter - 1] - E[iter]) / max(1, E[iter - 1])

    Hfinal = H[iter].T

    return Hfinal, E


def computeZi(X, A, B, n, g, f, maxiter_inner):
    iter = 1
    Z = {iter: np.random.rand(g, f)}
    Y = {iter: Z[iter]}
    AA = A.T @ A
    BB = B @ B.T

    alpha = {iter: 1}
    E = {iter: ((np.linalg.norm(X - (A @ Z[iter] @ B), 'fro')) ** 2) / n}

    L = np.linalg.norm(AA) * np.linalg.norm(BB)
    err = 1

    while iter <= maxiter_inner and err >= 1e-06:
        iter += 1
        GradHY = (AA @ Z[iter - 1] @ BB) - (A.T @ X @ B.T)
        Z[iter] = Y[iter - 1] - GradHY / L
        Z[iter][Z[iter] < 0] = 0
        alpha[iter] = (1 + np.sqrt((4 * alpha[iter - 1]) ** 2) + 1) / 2
        Y[iter] = Z[iter] + ((alpha[iter - 1] / alpha[iter]) * (Z[iter] - Z[iter - 1]))
        E[iter] = ((np.linalg.norm(X - (A @ Z[iter] @ B), 'fro')) ** 2) / n
        err = (E[iter - 1] - E[iter]) / max(1, E[iter - 1])

    Zfinal = Z[iter]

    return Zfinal, E


def constructA(Z, theta, p, r, g):
    P = np.eye(p)
    for i in range(1, g + 1):
        S = constructS(theta, r[i])
        P = P @ Z[i] @ S
    return P


def computeHm(X, A, r, n, maxiter_inner):
    iter = 1
    H = {iter: np.random.rand(r, n)}
    Y = {iter: H[iter]}
    AA = A.T @ A

    alpha = {iter: 1}
    E = {iter: ((np.linalg.norm(X - (A @ H[iter]), 'fro')) ** 2) / n}

    L = np.linalg.norm(AA)
    err = 1

    while (iter <= maxiter_inner and err >= 1e-06):
        iter += 1
        GradHY = (AA @ Y[iter - 1]) - (A.T @ X)
        H[iter] = Y[iter - 1] - GradHY / L
        H[iter][H[iter] < 0] = 0
        alpha[iter] = (1 + np.sqrt((4 * alpha[iter - 1]) ** 2) + 1) / 2
        Y[iter] = H[iter] + ((alpha[iter - 1] / alpha[iter]) * (H[iter] - H[iter - 1]))
        E[iter] = ((np.linalg.norm(X - (A @ H[iter]), 'fro')) ** 2) / n
        err = (E[iter - 1] - E[iter]) / max(1, E[iter - 1])

    Hfinal = H[iter]

    return Hfinal, E


def dnsNMF(X, theta, p, n, r, m, maxiter, maxiter_inner):
    Z, H = pretrain(X, m, r)
    iter = 1
    E = {}
    EE = {iter: (obj(X, Z, H, theta, r, m, p, n))[1]}
    while iter <= maxiter:
        start = 2
        final = m
        B = constructB(Z, H, theta, r, m, start, final)
        Z[1], _ = computeZ1(X, B, p, n, r[1], maxiter_inner)
        for i in range(2, m + 1):
            ##Compute Ai
            final = i - 1
            A = constructA(Z, theta, p, r, final)

            ## Compute Bi
            start = i + 1
            final = m
            B = constructB(Z, H, theta, r, m, start, final)

            ## Compute Zi
            Z[i], _ = computeZi(X, A, B, n, r[i - 1], r[i], maxiter_inner)

        iter = iter + 1

        ##Compute A
        final = m
        A = constructA(Z, theta, p, r, final)
        H[m], _ = computeHm(X, A, r[m], n, maxiter_inner)
        E[iter], EE[iter] = obj(X, Z, H, theta, r, m, p, n)

    final = m
    ZSfinal = constructA(Z, theta, p, r, final)
    Hfinal = H[m]
    return ZSfinal, Hfinal, E, EE


def run_model(l, theta_list, matImg, y, k1_list, k2_list, maxiter, maxiter_inner, maxiter_kmeans):

    # Normalise data
    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0

    # Definition of X
    X = normal_img.T
    m, n = X.shape

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    for k1 in k1_list:
        maxAcc = {}
        maxNmi = {}
        maxRecon_reeor = {}
        ##
        meanAcc = {}
        meanNmi = {}
        meanRecon_reeor = {}

        for k2 in k2_list:

            maxlst_acc = []
            maxlst_nmi = []
            maxlst_recon_err = []
            ##
            meanlst_acc = []
            meanlst_nmi = []
            meanlst_recon_err = []

            r = {}
            r[1] = k1
            r[2] = k2

            for theta in theta_list:
                Z, H, _, _ = dnsNMF(X, theta, m, n, r, l, maxiter, maxiter_inner)

                ## Reconstructed matix
                X_reconstructed = Z @ H

                ## Kmeans task
                lst_acc = []
                lst_nmi = []
                lst_recon_err = []

                for i in range(1, maxiter_kmeans):
                    pred = []
                    pred = kmeans.fit_predict(H.T)

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

            maxAcc[k2] = maxlst_acc
            maxNmi[k2] = maxlst_nmi
            maxRecon_reeor[k2] = maxlst_recon_err

            ##
            meanAcc[k2] = meanlst_acc
            meanNmi[k2] = meanlst_nmi
            meanRecon_reeor[k2] = meanlst_recon_err
            ## ENd for k2

        maxacc_final = {}
        maxnmi_final = {}
        maxrecon_final = {}

        print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs")
        for k_2 in k2_list:
            maxacc_final[k_2] = [max(maxAcc[k_2]), theta_list[np.argmax(maxAcc[k_2])]]
            maxnmi_final[k_2] = [max(maxNmi[k_2]), theta_list[np.argmax(maxNmi[k_2])]]
            maxrecon_final[k_2] = [max(maxRecon_reeor[k_2]), theta_list[np.argmax(maxRecon_reeor[k_2])]]
            print(f"##################################################################################################")
            print(f" k1 = {k1} : k2 = {k_2} ")
            print(f" ACC : {maxacc_final[k_2][0]}, with theta = {theta_list[np.argmax(maxAcc[k_2])]}")
            print(f" NMI : {maxnmi_final[k_2][0]}, with theta = {theta_list[np.argmax(maxNmi[k_2])]}")
            print(f" Reconstruction Error : {maxrecon_final[k_2][0]}")
            print(f"##################################################################################################")
        ##

        meanacc_final = {}
        meannmi_final = {}
        meanrecon_final = {}

        print("The results of running the Kmeans method 20 times and the average of 20 runs")
        for k_2 in k2_list:
            meanacc_final[k_2] = [max(meanAcc[k_2]), theta_list[np.argmax(meanAcc[k_2])]]
            meannmi_final[k_2] = [max(meanNmi[k_2]), theta_list[np.argmax(meanNmi[k_2])]]
            meanrecon_final[k_2] = [max(meanRecon_reeor[k_2]), theta_list[np.argmax(meanRecon_reeor[k_2])]]
            print(f"##################################################################################################")
            print(f" k1 = {k1} : k2 = {k_2}")
            print(f" ACC : {meanacc_final[k_2][0]}, with theta = {theta_list[np.argmax(meanAcc[k_2])]}")
            print(f" NMI : {meannmi_final[k_2][0]}, with theta = {theta_list[np.argmax(meanNmi[k_2])]}")
            print(f" Reconstruction Error : {meanrecon_final[k_2][0]}")
            print(f"##################################################################################################")
    ##**
    print("done")

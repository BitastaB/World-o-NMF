""" Same model of dnsNMF with layer 1. Since it produces same results but much more faster than original nsNMF """

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


def nsNMF(X, theta, p, n, r, m, maxiter, maxiter_inner):
    Z, H = pretrain(X, m, r)
    iter = 1
    E = {}
    EE = {iter: (obj(X, Z, H, theta, r, m, p, n))[1]}
    err = 1
    while iter <= maxiter and err >= 1e-06:
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
        err = (E[iter - 1] - E[iter]) / max(1, E[iter - 1])

    final = m
    ZSfinal = constructA(Z, theta, p, r, final)
    Hfinal = H[m]
    return ZSfinal, Hfinal, E, EE


def run_model(theta_list, matImg, y, k_list, maxiter, maxiter_inner, maxiter_kmeans):
    l = 1
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

    for k in k_list:

        max_lst_acc_k = []
        max_lst_nmi_k = []
        max_lst_recon_err_k = []

        avg_lst_acc_k = []
        avg_lst_nmi_k = []
        avg_lst_recon_err_k = []

        r = {1: k}
        for theta in theta_list:
            Z, H, _, _ = nsNMF(X, theta, m, n, r, l, maxiter, maxiter_inner)

            ## Reconstructed matix
            X_reconstructed = Z @ H

            ## Kmeans task
            lst_acc = []
            lst_nmi = []

            for i in range(1, maxiter_kmeans):
                pred = kmeans.fit_predict(H.T)

                ## NMI
                nmi = 100 * evaluate_nmi(y, pred)

                ## ACC
                acc = 100 * accuracy(y, pred)

                lst_acc.append(acc)
                lst_nmi.append(nmi)
                ## End for

            ## Add max values to list for one theta
            max_lst_acc_k.append(max(lst_acc))
            max_lst_nmi_k.append(max(lst_nmi))

            ## Add avg values to list for one theta
            avg_lst_acc_k.append(statistics.mean(lst_acc))
            avg_lst_nmi_k.append(statistics.mean(lst_nmi))

        print(
            "**********************************************************************************************************")
        print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs\n")
        print(f"k = {k} : best max_acc = {max(max_lst_acc_k)} , with nu = {theta_list[np.argmax(max_lst_acc_k)]}")
        print(f"k = {k} : best max_nmi = {max(max_lst_nmi_k)} , with nu = {theta_list[np.argmax(max_lst_acc_k)]}")
        print("\n\nThe results of running the Kmeans method 20 times and the report of average of 20 runs\n")
        print(f"k = {k} : best avg_acc = {max(avg_lst_acc_k)} , with nu = {theta_list[np.argmax(avg_lst_acc_k)]} ")
        print(f"k = {k} : best avg_nmi = {max(avg_lst_nmi_k)} , with nu = {theta_list[np.argmax(avg_lst_nmi_k)]} ")
        print(
            "**********************************************************************************************************")

    print("done")

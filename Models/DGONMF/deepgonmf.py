import numpy as np
from sklearn.decomposition import NMF
import statistics
from Utils.metrics_evaluation import accuracy, evaluate_nmi
from Utils.utils import init_kmeans, construct_similarity_matrix


"""Upadte H_i and V_i for i=1,2,...,l"""


# Update H_i if i = 1
def update_H_1(matX, dictH, dictV, phi, sci, myeps_1, myeps_2, l):
    num = matX @ dictV[l].T @ sci[2].T
    den = dictH[1] @ sci[2] @ dictV[l] @ dictV[l].T @ sci[2].T
    den[den < myeps_1] = myeps_2
    res = np.divide(num, den)
    dictH[1] = np.multiply(dictH[1], np.sqrt(res))
    return dictH


# Update H_i when i>1
def update_H(i, matX, dictH, dictV, phi, sci, myeps_1, myeps_2, l):
    num = phi[i - 1].T @ matX @ dictV[l].T @ sci[i + 1].T
    den = phi[i - 1].T @ phi[i - 1] @ dictH[i] @ sci[i + 1] @ dictV[l] @ dictV[l].T @ sci[i + 1].T
    den[den < myeps_1] = myeps_2
    res = num / den
    dictH[i] = np.multiply(dictH[i], np.sqrt(res))
    return dictH


# Update V_i if i == l
def update_V_l(matX, dictH, dictV, phi, sci, matS, matD, alpha, beta, myeps_1, myeps_2, r_i, l):  ##alpha and beta???
    num = (phi[l].T @ matX) + (alpha * dictV[l] @ matS) + (beta * dictV[l])

    ones = np.ones((r_i, r_i))
    den = (phi[l].T @ phi[l] @ dictV[l]) + (alpha * dictV[l] @ matD) + (beta * ones @ dictV[l])

    den[den < myeps_1] = myeps_2
    res = num / den
    dictV[l] = np.multiply(dictV[l], np.sqrt(res))
    return dictV


# Update V_i when i<l
def update_V(i, matX, dictH, dictV, phi, sci, matS, matD, alpha, beta, myeps_1, myeps_2, r_i, l):
    num = (phi[i].T @ matX) + (alpha * dictV[i] @ matS) + (beta * dictV[i])
    ones = np.ones((r_i, r_i))
    den = (phi[i].T @ phi[i] @ dictV[i]) + (alpha * dictV[i] @ matD) + (beta * ones @ dictV[i])

    den[den < myeps_1] = myeps_2
    res = num / den
    dictV[i] = np.multiply(dictV[i], np.sqrt(res))
    return dictV


"""Deep Graph Orthogonal NMF (DGONMF)"""


def DGONMF(matX, matS, matD, m, n, l, k, alpha, beta, max_iter, myeps_1, myeps_2):
    ## Pretrain all layers
    dictH = {}
    dictV = {}
    matZ = matX
    for i in range(1, l + 1):
        r_i = k[i - 1]
        nmf_model = NMF(r_i)
        nmf_features = nmf_model.fit_transform(matZ)  ##H
        dictH[i] = nmf_features
        nmf_components = nmf_model.components_  ##V
        dictV[i] = nmf_components
        matZ = dictV[i]

    matH = np.identity(dictH[1].shape[0])
    for q in range(1, l + 1):
        matH = matH @ dictH[q]

    # Calculate final V'
    matV = dictV
    matX_reconstructed = matH @ dictV[l]
    ae = np.linalg.norm(matX - matX_reconstructed, 'fro')
    errvec = {0: (ae ** 2) / n}

    ### Fine-tune all layers
    err = 1
    t = 1

    while t <= max_iter and err >= 1e-04:
        phi = {}
        sci = {}

        for i in range(1, l + 1):

            # Define phi_i-1
            if i == 1:
                phi[0] = np.identity(
                    m)
            else:
                temp = np.identity(dictH[1].shape[0])
                for j in range(1, i):
                    temp = temp @ dictH[j]
                phi[i - 1] = temp

            # Define sci_i+1

            if i == l:
                sci[i + 1] = np.identity(k[i - 1])
            else:
                sci[i + 1] = dictH[i + 1]
                for j in range(i + 2, l + 1):
                    sci[i + 1] = np.matmul(sci[i + 1], dictH[j])

                    # Update H_i
            if i == 1:
                dictH = update_H_1(matX, dictH, dictV, phi, sci, myeps_1, myeps_2, l)
            else:
                dictH = update_H(i, matX, dictH, dictV, phi, sci, myeps_1, myeps_2, l)

            # Update phi_i
            phi[i] = phi[i - 1] @ dictH[i]

            r_i = k[i - 1]
            # Update V_i
            if i == l:
                dictV = update_V_l(matX, dictH, dictV, phi, sci, matS, matD, alpha, beta, myeps_1, myeps_2, r_i, l)
                matH = np.identity(dictH[1].shape[0])
                for q in range(1, l + 1):
                    matH = matH @ dictH[q]

                # Calculate final V'
                matV = dictV
                matX_reconstructed = matH @ dictV[l]
                ae = np.linalg.norm(matX - matX_reconstructed, 'fro')
                errvec[t] = (ae ** 2) / n
                err = (errvec[t - 1] - errvec[t]) / max(1, errvec[t - 1])
                t += 1
            else:
                dictV = update_V(i, matX, dictH, dictV, phi, sci, matS, matD, alpha, beta, myeps_1, myeps_2, r_i, l)

    # Calculate final H'
    matH = np.identity(dictH[1].shape[0])
    for q in range(1, l + 1):
        matH = matH @ dictH[q]
    # Calculate final V'
    matV = dictV[l]
    return matH, matV


def run_model(alpha_range, beta_range, matImg, matGnd, k_1_list, k_2_list, maxiter_kmeans, l, max_iter,
              myeps_1, myeps_2, y):

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    # Normalise data
    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0

    # Definition of X
    matX = normal_img.T
    m, n = matX.shape

    parameters = []
    for a in alpha_range:
        for b in beta_range:
            parameters.append((a, b))

    # Create similarity matrix
    matS = construct_similarity_matrix(matGnd)

    # Initialization
    diag = np.sum(matS, axis=1)
    matD = np.diag(diag)

    for k_1 in k_1_list:
        maxAcc = {}
        maxNmi = {}
        maxRecon_reeor = {}
        ##
        meanAcc = {}
        meanNmi = {}
        meanRecon_reeor = {}

        for k_2 in k_2_list:  # range(sk, ek, 10):
            maxlst_acc = []
            maxlst_nmi = []
            maxlst_recon_err = []
            ##
            meanlst_acc = []
            meanlst_nmi = []
            meanlst_recon_err = []
            for p in range(len(parameters)):
                alpha = parameters[p][0]
                beta = parameters[p][1]

                # run deep gonmf for max_iter times
                matH, matV = DGONMF(matX, matS, matD, m, n, l, [k_1, k_2], alpha, beta, max_iter, myeps_1, myeps_2)

                # Reconstructed matrix
                matX_reconstructed = matH @ matV

                # Kmeans task
                lst_acc = []
                lst_nmi = []
                lst_recon_err = []

                for i in range(1, maxiter_kmeans):
                    pred = []
                    pred = kmeans.fit_predict(matV.T)

                    # NMI
                    nmi = 100 * evaluate_nmi(y, pred)

                    # ACC
                    acc = 100 * accuracy(y, pred)

                    # Reconstruction Error
                    a = np.linalg.norm(matX - matX_reconstructed, 'fro')
                    recon_reeor = (a)  # /n

                    lst_acc.append(acc)
                    lst_nmi.append(nmi)
                    lst_recon_err.append(recon_reeor)
                    # End for

                ##
                maxlst_acc.append(max(lst_acc))
                maxlst_nmi.append(max(lst_nmi))
                maxlst_recon_err.append(max(lst_recon_err))

                ##
                meanlst_acc.append(statistics.mean(lst_acc))
                meanlst_nmi.append(statistics.mean(lst_nmi))
                meanlst_recon_err.append(statistics.mean(lst_recon_err))
            # End for

            maxAcc[k_2] = maxlst_acc
            maxNmi[k_2] = maxlst_nmi
            maxRecon_reeor[k_2] = maxlst_recon_err

            ##
            meanAcc[k_2] = meanlst_acc
            meanNmi[k_2] = meanlst_nmi
            meanRecon_reeor[k_2] = meanlst_recon_err
        # ENd for k2

        maxacc_final = {}
        maxnmi_final = {}
        maxrecon_final = {}

        print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs")
        for k_2 in k_2_list:
            maxacc_final[k_2] = [max(maxAcc[k_2]), parameters[np.argmax(maxAcc[k_2])]]
            maxnmi_final[k_2] = [max(maxNmi[k_2]), parameters[np.argmax(maxNmi[k_2])]]
            maxrecon_final[k_2] = [max(maxRecon_reeor[k_2]), parameters[np.argmax(maxRecon_reeor[k_2])]]
            print(f"##################################################################################################")
            print(f" k1 = {k_1} : k2 = {k_2} ")
            print(f" Max ACC : {maxacc_final[k_2][0]}, with (alpha, beta) = {parameters[np.argmax(maxAcc[k_2])]}")
            print(f" Max NMI : {maxnmi_final[k_2][0]}, with (alpha, beta) = {parameters[np.argmax(maxNmi[k_2])]}")
            print(f" Reconstruction Error : {maxrecon_final[k_2][0]}")
            print(f"##################################################################################################")
        ##

        meanacc_final = {}
        meannmi_final = {}
        meanrecon_final = {}

        print("The results of running the Kmeans method 20 times and the average of 20 runs")
        for k_2 in k_2_list:
            meanacc_final[k_2] = [max(meanAcc[k_2]), parameters[np.argmax(meanAcc[k_2])]]
            meannmi_final[k_2] = [max(meanNmi[k_2]), parameters[np.argmax(meanNmi[k_2])]]
            meanrecon_final[k_2] = [max(meanRecon_reeor[k_2]), parameters[np.argmax(meanRecon_reeor[k_2])]]
            print(f"##################################################################################################")
            print(f" k1 = {k_1} : k2 = {k_2}")
            print(f" Avg ACC : {meanacc_final[k_2][0]}, with (alpha, beta) = {parameters[np.argmax(meanAcc[k_2])]}")
            print(f" Avg NMI : {meannmi_final[k_2][0]}, with (alpha, beta) = {parameters[np.argmax(meanNmi[k_2])]}")
            print(f" Reconstruction Error : {meanrecon_final[k_2][0]}")
            print(f"##################################################################################################")

    print("#Done!")

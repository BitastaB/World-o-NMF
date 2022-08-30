import numpy as np
import statistics
from sklearn.decomposition import NMF
from Utils.metrics_evaluation import evaluate_nmi, accuracy
from Utils.utils import init_kmeans


def run_model(matImg, y, k_list, maxiter_kmeans):

    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    ## Definiton of matX
    matX = normal_img.T

    for k in k_list:
        model = NMF(n_components=k, init='random')
        W = model.fit_transform(matX)
        H = model.components_

        lst_acc = []
        lst_nmi = []
        for i in range(1, maxiter_kmeans):
            pred = []
            pred = kmeans.fit_predict(H.T)
            nm = 100 * evaluate_nmi(y, pred)
            ac = 100 * accuracy(y, pred)
            lst_acc.append(ac)
            lst_nmi.append(nm)

        maxacc = max(lst_acc)
        maxnmi = max(lst_nmi)

        meanacc = statistics.mean(lst_acc)
        meannmi = statistics.mean(lst_nmi)

        print(f"##################################################################################################")
        print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs")
        print(f"##################################################################################################")
        print(f" k = {k} : maxACC = {maxacc}")
        print(f" k = {k} : maxNMI = {maxnmi}")
        print(f"##################################################################################################")
        print("The results of running the Kmeans method 20 times and the average of 20 runs")
        print(f"##################################################################################################")
        print(f" k = {k} : meanACC = {meanacc}")
        print(f" k = {k} : meanNMI = {meannmi}")
        print("\n")

    print("#Done!")

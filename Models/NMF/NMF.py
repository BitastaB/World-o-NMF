import numpy as np
import statistics
from sklearn.decomposition import NMF
from Utils.metrics_evaluation import evaluate_nmi, accuracy, calculate_silhouette_score, calculate_davies_bouldin_score, \
    calculate_dunn_index
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
        lst_sil_score = []
        lst_davis_score = []
        lst_dunn_score = []

        for i in range(1, maxiter_kmeans):
            pred = []
            pred = kmeans.fit_predict(H.T)
            nm = 100 * evaluate_nmi(y, pred)
            ac = 100 * accuracy(y, pred)

            # Silhoutte score
            silhouette_score = calculate_silhouette_score(H.T, pred)

            # Davis-bouldin score
            davis_score = calculate_davies_bouldin_score(H.T, pred)

            # dunn's index
            dunn_score = calculate_dunn_index(H.T, y)

            lst_acc.append(ac)
            lst_nmi.append(nm)
            lst_sil_score.append(silhouette_score)
            lst_davis_score.append(davis_score)
            lst_dunn_score.append(dunn_score)

        maxacc = max(lst_acc)
        maxnmi = max(lst_nmi)
        maxSilScore = max(lst_sil_score)
        maxDunnScore = max(lst_dunn_score)
        minDaviesScore = min(lst_davis_score)

        meanacc = statistics.mean(lst_acc)
        meannmi = statistics.mean(lst_nmi)
        meanSilScore = statistics.mean(lst_sil_score)
        meanDunnScore = statistics.mean(lst_dunn_score)
        meanDaviesScore = statistics.mean(lst_davis_score)

        print(f"##################################################################################################")
        print("The results of running the Kmeans method 20 times and the report of maximum of 20 runs")
        print(f"##################################################################################################")
        print(f" k = {k} : max ACC = {maxacc}")
        print(f" k = {k} : max NMI = {maxnmi}")
        print(f" k = {k} : max Silhoutte Score = {maxSilScore}")
        print(f" k = {k} : max Dunn's Index Score = {maxDunnScore}")
        print(f" k = {k} : min Davies bouldin Score = {minDaviesScore}")

        print(f"##################################################################################################")
        print("The results of running the Kmeans method 20 times and the average of 20 runs")
        print(f"##################################################################################################")
        print(f" k = {k} : mean ACC = {meanacc}")
        print(f" k = {k} : mean NMI = {meannmi}")
        print(f" k = {k} : mean Silhoutte score = {meanSilScore}")
        print(f" k = {k} : mean Dunn's Index Score = {meanDunnScore}")
        print(f" k = {k} : mean Davies bouldin Score = {meanDaviesScore}")

        print("\n")

    print("#Done!")

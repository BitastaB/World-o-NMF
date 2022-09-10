import numpy as np
import statistics
from sklearn.decomposition import NMF
from Utils.metrics_evaluation import evaluate_nmi, accuracy, calculate_silhouette_score, calculate_davies_bouldin_score, \
    calculate_dunn_index
from Utils.utils import init_kmeans, store_kmeans


def run_model(model, dataset, matImg, y, k_list, maxiter_kmeans):

    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0

    # Initialise Kmeans
    kmeans = init_kmeans(y)

    ## Definiton of matX
    matX = normal_img.T

    # Util for plotting best cluster produced
    best_cluster_acc = {'acc': 0}

    for k in k_list:
        model_nmf = NMF(n_components=k, init='random')
        W = model_nmf.fit_transform(matX)
        H = model_nmf.components_

        lst_acc = []
        lst_nmi = []
        lst_sil_score = []
        lst_davis_score = []
        lst_dunn_score = []

        for i in range(1, maxiter_kmeans):
            pred = kmeans.fit_predict(H.T)
            nmi = 100 * evaluate_nmi(y, pred)
            acc = 100 * accuracy(y, pred)
            if acc > best_cluster_acc['acc']:
                best_cluster_acc['acc'] = acc
                best_cluster_acc['data'] = H
                best_cluster_acc['pred'] = pred

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


    # Storing details of best cluster
    data = best_cluster_acc['data']
    pred = best_cluster_acc['pred']
    store_kmeans(data, pred, model, dataset)

    print("#Done!")

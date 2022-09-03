import numpy as np

from Utils.plot_metrics import plot_clusters_tsne, plot_acc, plot_nmi, plot_sil, plot_dunn, plot_db, \
    plot_model_performance
from Utils.plot_utils import DGONMF_lists, erwnmf_lists, nmf_lists, gnmf_lists, dnsNMF_lists, dsnmf_lists, OGNMF_lists, \
    GRSNFM_lists, RSCNMF_lists, nsNMF_lists, DGRSNMF_lists


def plot_acc_nmi(dataset):
    round_upto = 4
    k_list = [10, 20, 30, 40, 50, 60, 70]

    # Get average metric values for models
    dgonmf_lst = DGONMF_lists(dataset, round_upto)
    erwnmf_lst = erwnmf_lists(dataset, round_upto)
    nmf_lst = nmf_lists(dataset, round_upto)
    gnmf_lst = gnmf_lists(dataset, round_upto)
    dnsNMF_lst = dnsNMF_lists(dataset, round_upto)
    dsnmf_lst = dsnmf_lists(dataset, round_upto)
    ognmf_lst = OGNMF_lists(dataset, round_upto)
    grsnmf_lst = GRSNFM_lists(dataset, round_upto)
    rscnmf_lst = RSCNMF_lists(dataset, round_upto)
    nsnmf_lst = nsNMF_lists(dataset, round_upto)
    dgrsnmf_lst = DGRSNMF_lists(dataset, round_upto)

    plot_acc(dataset, k_list, dgonmf_lst[0], erwnmf_lst[0], nmf_lst[0], gnmf_lst[0], dnsNMF_lst[0],
             dsnmf_lst[0], ognmf_lst[0], grsnmf_lst[0], rscnmf_lst[0], nsnmf_lst[0], dgrsnmf_lst[0])

    plot_nmi(dataset, k_list, dgonmf_lst[1], erwnmf_lst[1], nmf_lst[1], gnmf_lst[1], dnsNMF_lst[1],
            dsnmf_lst[1], ognmf_lst[1], grsnmf_lst[1], rscnmf_lst[1], nsnmf_lst[1], dgrsnmf_lst[1])

    plot_sil(dataset, k_list, dgonmf_lst[2], erwnmf_lst[2], nmf_lst[2], gnmf_lst[2], dnsNMF_lst[2],
             dsnmf_lst[2], ognmf_lst[2], grsnmf_lst[2], rscnmf_lst[2], nsnmf_lst[2], dgrsnmf_lst[2])

    plot_dunn(dataset, k_list, dgonmf_lst[3], erwnmf_lst[3], nmf_lst[3], gnmf_lst[3], dnsNMF_lst[3],
              dsnmf_lst[3], ognmf_lst[3], grsnmf_lst[3], rscnmf_lst[3], nsnmf_lst[3], dgrsnmf_lst[3])

    plot_db(dataset, k_list, dgonmf_lst[4], erwnmf_lst[4], nmf_lst[4], gnmf_lst[4], dnsNMF_lst[4],
            dsnmf_lst[4], ognmf_lst[4], grsnmf_lst[4], rscnmf_lst[4], nsnmf_lst[4], dgrsnmf_lst[4])
def plot_clusters(model, dataset):
    path = f'./Results/{dataset}/kmeans_{model}_{dataset}.npz'
    file = np.load(path)
    data = file['data']
    pred = file['kmneans_pred']
    n_cluster = np.unique(pred).shape[0]
    plot_clusters_tsne(data.T, model, dataset, kmeans_cluster=n_cluster)


if __name__ == '__main__':
    dataset = "umist"  # [jaffe, orl, warpAR10P, umist, yale, yaleB]
    model = "DGRSNMF" #[DGONMF, dnsNMF, dsnmf, ERWNMF, RSCNMF, OGNMF, GRSNMF, GNMF, NMF, nsNMF, DGRSNMF]

   # plot_acc_nmi(dataset)
   # plot_model_performance("DGONMF")
    for i in range(10):
        plot_clusters(model, dataset)

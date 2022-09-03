import numpy as np
from bokeh.palettes import Category10, Category20b
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import pandas as pd

from Utils.plot_utils import DGONMF_lists

pallet = {0: '#090119',
          1: '#1A7184',
          2: '#DAFF1A',
          3: '#60FC89',
          4: '#C90E6A',
          5: '#7E9D39',
          6: '#F66F14',
          7: '#EFA2BF',
          8: '#AAF846',
          9: '#1FE5B7',
          10: '#782D38',
          11: '#739CEF',
          12: '#41327B',
          13: '#7D039F',
          14: '#2135F1',
          15: '#CFC9F2',
          16: '#F633EE',
          17: '#F77F76',
          18: '#C49BCB',
          19: '#1EBF71',
          20: '#A7D2D2',
          21: '#FADAC3',
          22: '#A7A298',
          23: '#F6E05B',
          24: '#AA75A5',
          25: '#A8845C',
          26: '#F8FF80',
          27: '#FA607E',
          28: '#F02B0E',
          29: '#CEAE55',
          30: '#C67A07',
          31: '#D8B189',
          32: '#3F8701',
          33: '#751DF7',
          34: '#7ACB31',
          35: '#DCA142',
          36: '#B93719',
          37: '#2FE1FB',
          38: '#F6358A',
          39: '#FFBF00'}


def plot_clusters_tsne(data, model, dataset, kmeans_cluster):
    perp_list = [12]# [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50]

    for perp in perp_list:
        _title = f"dataset = {dataset}, model = {model}, with tsne perpexlity = {perp}"

        tsne = TSNE(n_components=3, verbose=1, perplexity=perp, n_iter=5000, learning_rate=250)
        tsne_scale_results = tsne.fit_transform(data)
        tsne_df_scale = pd.DataFrame(tsne_scale_results, columns=['tsne1', 'tsne2', 'tsne3'])

        kmeans_tsne_scale = KMeans(n_clusters=kmeans_cluster, n_init=100, max_iter=400, init='k-means++',
                                   random_state=42).fit(tsne_df_scale)

        labels_tsne_scale = kmeans_tsne_scale.labels_
        clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters': labels_tsne_scale})], axis=1)

        # plt.figure(figsize = (15,15))
        sns.scatterplot(clusters_tsne_scale.iloc[:, 0], clusters_tsne_scale.iloc[:, 1], hue=labels_tsne_scale,
                        palette=pallet, s=50, alpha=0.75).set_title(_title)  # , fontsize=15)
        plt.legend([], [], frameon=False)
        plt.savefig(f'Results/plots/{dataset}/cluster_{dataset}_{model}.png', dpi=150)
        plt.show()


def plot_model_performance(model):
    round_upto = 4
    k_list = [10, 20, 30, 40, 50, 60, 70]
    jaffe_acc_list, jaffe_nmi_list = DGONMF_lists("jaffe", round_upto)
    orl_acc_list, orl_nmi_list = DGONMF_lists("orl", round_upto)
    warp_acc_list, warp_nmi_list = DGONMF_lists("warpAR10P", round_upto)
    umist_acc_list, umist_nmi_list = DGONMF_lists("umist", round_upto)
    yale_acc_list, yale_nmi_list = DGONMF_lists("yale", round_upto)

    # Plotting average accuracy
    plt.plot(k_list, jaffe_acc_list, label="jaffe", marker='o')
    plt.plot(k_list, orl_acc_list, label="orl", marker='o')
    plt.plot(k_list, warp_acc_list, label="warpAR10P", marker='o')
    plt.plot(k_list, umist_acc_list, label="umist", marker='o')
    plt.plot(k_list, yale_acc_list, label="yale", marker='o')

    plt.xlabel("k")
    plt.ylabel("accurancy in %")
    plt.legend()
    plt.title(f"Average accuracy for : {model}")
    plt.savefig(f'Results/plots/{model}_acc.png', dpi=150)
    plt.show()

    # Plotting average nmi
    plt.plot(k_list, jaffe_nmi_list, label="jaffe", marker='o')
    plt.plot(k_list, orl_nmi_list, label="orl", marker='o')
    plt.plot(k_list, warp_nmi_list, label="warpAR10P", marker='o')
    plt.plot(k_list, umist_nmi_list, label="umist", marker='o')
    plt.plot(k_list, yale_nmi_list, label="yale", marker='o')

    plt.xlabel("k")
    plt.ylabel("NMI in %")
    plt.legend()
    plt.title(f"Average NMI for : {model}")
    plt.savefig(f'Results/plots/{model}_nmi.png', dpi=150)
    plt.show()


# Plotting average accuracy
def plot_acc(dataset, k_list, dgonmf_acc_list,erwnmf_acc_list, nmf_acc_list, gnmf_acc_list, dnsNMF_acc_list, dsNMF_acc_list, ognmf_acc_list, grsnmf_acc_list, rscnmf_acc_list, nsnmf_acc_list, dgrsnmf_acc_list):
    # Plotting average accuracy
    plt.plot(k_list, dgonmf_acc_list, label="DGONMF", marker='.', c=Category10[10][0])
    plt.plot(k_list, erwnmf_acc_list, label="ERWNMF", marker='.', c=Category10[10][1])
    plt.plot(k_list, nmf_acc_list, label="NMF", marker='.', c=Category10[10][2])
    plt.plot(k_list, gnmf_acc_list, label="GNMF", marker='.', c=Category10[10][3])
    plt.plot(k_list, dnsNMF_acc_list, label="dnsNMF", marker='.', c=Category10[10][4])
    plt.plot(k_list, dsNMF_acc_list, label="dsnmf", marker='.', c=Category10[10][5])
    plt.plot(k_list, ognmf_acc_list, label="OGNMF", marker='.', c=Category10[10][6])
    plt.plot(k_list, grsnmf_acc_list, label="GRSNMF", marker='.', c=Category10[10][7])
    plt.plot(k_list, rscnmf_acc_list, label="RSCNMF", marker='.', c=Category10[10][8])
    plt.plot(k_list, nsnmf_acc_list, label="nsNMF", marker='.', c=Category10[10][9])
    plt.plot(k_list, dgrsnmf_acc_list, label="DGRSNMF", marker='.', c=Category20b[12][0])

    plt.xlabel("k")
    plt.ylabel("accurancy in %")
    plt.legend(loc="right")
    plt.title(f"Average accuracy for : {dataset}")
    plt.savefig(f'Results/plots/{dataset}/{dataset}_acc.png', dpi=150)
    plt.show()

def plot_nmi(dataset, k_list, dgonmf_nmi_list, erwnmf_nmi_list, nmf_nmi_list, gnmf_nmi_list, dnsNMF_nmi_list, dsNMF_nmi_list, ognmf_nmi_list, grsnmf_nmi_list, rscnmf_nmi_list, nsnmf_nmi_list, dgrsnmf_nmi_list):
    plt.plot(k_list, dgonmf_nmi_list, label="DGONMF", marker='.', c=Category10[10][0])
    plt.plot(k_list, erwnmf_nmi_list, label="ERWNMF", marker='.', c=Category10[10][1])
    plt.plot(k_list, nmf_nmi_list, label="NMF", marker='.', c=Category10[10][2])
    plt.plot(k_list, gnmf_nmi_list, label="GNMF", marker='.', c=Category10[10][3])
    plt.plot(k_list, dnsNMF_nmi_list, label="dnsNMF", marker='.', c=Category10[10][4])
    plt.plot(k_list, dsNMF_nmi_list, label="dsnmf", marker='.', c=Category10[10][5])
    plt.plot(k_list, ognmf_nmi_list, label="OGNMF", marker='.', c=Category10[10][6])
    plt.plot(k_list, grsnmf_nmi_list, label="GRSNMF", marker='.', c=Category10[10][7])
    plt.plot(k_list, rscnmf_nmi_list, label="RSCNMF", marker='.', c=Category10[10][8])
    plt.plot(k_list, nsnmf_nmi_list, label="nsNMF", marker='.', c=Category10[10][9])
    plt.plot(k_list, dgrsnmf_nmi_list, label="DGRSNMF", marker='.', c=Category20b[12][0])

    plt.xlabel("k")
    plt.ylabel("nmi in %")
    plt.legend(loc="right")
    plt.title(f"Average nmi for : {dataset}")
    plt.savefig(f'Results/plots/{dataset}/{dataset}_nmi.png', dpi=150)
    plt.show()

def plot_sil(dataset, k_list, dgonmf_sil_list, erwnmf_sil_list, nmf_sil_list, gnmf_sil_list, dnsNMF_sil_list, dsNMF_sil_list, ognmf_sil_list, grsnmf_sil_list, rscnmf_sil_list, nsnmf_sil_list, dgrsnmf_sil_list):
    plt.plot(k_list, dgonmf_sil_list, label="DGONMF", marker='.', c=Category10[10][0])
    plt.plot(k_list, erwnmf_sil_list, label="ERWNMF", marker='.', c=Category10[10][1])
    plt.plot(k_list, nmf_sil_list, label="NMF", marker='.', c=Category10[10][2])
    plt.plot(k_list, gnmf_sil_list, label="GNMF", marker='.', c=Category10[10][3])
    plt.plot(k_list, dnsNMF_sil_list, label="dnsNMF", marker='.', c=Category10[10][4])
    plt.plot(k_list, dsNMF_sil_list, label="dsnmf", marker='.', c=Category10[10][5])
    plt.plot(k_list, ognmf_sil_list, label="OGNMF", marker='.', c=Category10[10][6])
    plt.plot(k_list, grsnmf_sil_list, label="GRSNMF", marker='.', c=Category10[10][7])
    plt.plot(k_list, rscnmf_sil_list, label="RSCNMF", marker='.', c=Category10[10][8])
    plt.plot(k_list, nsnmf_sil_list, label="nsNMF", marker='.', c=Category10[10][9])
    plt.plot(k_list, dgrsnmf_sil_list, label="DGRSNMF", marker='.', c=Category20b[12][0])

    plt.xlabel("k")
    plt.ylabel("sil in %")
    plt.legend(loc="right")
    plt.title(f"Average Silhoutter Score for : {dataset}")
    plt.savefig(f'Results/plots/{dataset}/{dataset}_sil.png', dpi=150)
    plt.show()


def plot_dunn(dataset, k_list, dgonmf_dunn_list, erwnmf_dunn_list, nmf_dunn_list, gnmf_dunn_list, dnsNMF_dunn_list, dsNMF_dunn_list, ognmf_dunn_list, grsnmf_dunn_list, rscnmf_dunn_list, nsnmf_dunn_list, dgrsnmf_dunn_list):
    plt.plot(k_list, dgonmf_dunn_list, label="DGONMF", marker='.', c=Category10[10][0])
    plt.plot(k_list, erwnmf_dunn_list, label="ERWNMF", marker='.', c=Category10[10][1])
    plt.plot(k_list, nmf_dunn_list, label="NMF", marker='.', c=Category10[10][2])
    plt.plot(k_list, gnmf_dunn_list, label="GNMF", marker='.', c=Category10[10][3])
    plt.plot(k_list, dnsNMF_dunn_list, label="dnsNMF", marker='.', c=Category10[10][4])
    plt.plot(k_list, dsNMF_dunn_list, label="dsnmf", marker='.', c=Category10[10][5])
    plt.plot(k_list, ognmf_dunn_list, label="OGNMF", marker='.', c=Category10[10][6])
    plt.plot(k_list, grsnmf_dunn_list, label="GRSNMF", marker='.', c=Category10[10][7])
    plt.plot(k_list, rscnmf_dunn_list, label="RSCNMF", marker='.', c=Category10[10][8])
    plt.plot(k_list, nsnmf_dunn_list, label="nsNMF", marker='.', c=Category10[10][9])
    plt.plot(k_list, dgrsnmf_dunn_list, label="DGRSNMF", marker='.', c=Category20b[12][0])

    plt.xlabel("k")
    plt.ylabel("dunn in %")
    plt.legend(loc="right")
    plt.title(f"Average Dunn's Index for : {dataset}")
    plt.savefig(f'Results/plots/{dataset}/{dataset}_dunn.png', dpi=150)
    plt.show()


def plot_db(dataset, k_list, dgonmf_db_list, erwnmf_db_list, nmf_db_list, gnmf_db_list, dnsNMF_db_list, dsNMF_db_list, ognmf_db_list, grsnmf_db_list, rscnmf_db_list, nsnmf_db_list, dgrsnmf_db_list):
    plt.plot(k_list, dgonmf_db_list, label="DGONMF", marker='.', c=Category10[10][0])
    plt.plot(k_list, erwnmf_db_list, label="ERWNMF", marker='.', c=Category10[10][1])
    plt.plot(k_list, nmf_db_list, label="NMF", marker='.', c=Category10[10][2])
    plt.plot(k_list, gnmf_db_list, label="GNMF", marker='.', c=Category10[10][3])
    plt.plot(k_list, dnsNMF_db_list, label="dnsNMF", marker='.', c=Category10[10][4])
    plt.plot(k_list, dsNMF_db_list, label="dsnmf", marker='.', c=Category10[10][5])
    plt.plot(k_list, ognmf_db_list, label="OGNMF", marker='.', c=Category10[10][6])
    plt.plot(k_list, grsnmf_db_list, label="GRSNMF", marker='.', c=Category10[10][7])
    plt.plot(k_list, rscnmf_db_list, label="RSCNMF", marker='.', c=Category10[10][8])
    plt.plot(k_list, nsnmf_db_list, label="nsNMF", marker='.', c=Category10[10][9])
    plt.plot(k_list, dgrsnmf_db_list, label="DGRSNMF", marker='.', c=Category20b[12][0])

    plt.xlabel("k")
    plt.ylabel("db in %")
    plt.legend(loc="right")
    plt.title(f"Average Davies Bouldin Index for : {dataset}")
    plt.savefig(f'Results/plots/{dataset}/{dataset}_db.png', dpi=150)
    plt.show()


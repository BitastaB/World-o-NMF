from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import pandas as pd

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


def plot_clusters(data, title, kmeans_cluster):
    perp_list = [15]

    for perp in perp_list:
        _title = title + f" tsne perpexlity = {perp}"

        tsne = TSNE(n_components=3, verbose=1, perplexity=perp, n_iter=5000, learning_rate=250)
        tsne_scale_results = tsne.fit_transform(data)
        tsne_df_scale = pd.DataFrame(tsne_scale_results, columns=['tsne1', 'tsne2', 'tsne3'])

        kmeans_tsne_scale = KMeans(n_clusters=kmeans_cluster, n_init=100, max_iter=400, init='k-means++',
                                   random_state=42).fit(tsne_df_scale)
        # kmeans_tsne_scale = pred
        print('KMeans tSNE Scaled Silhouette Score: {}'.format(
            silhouette_score(tsne_df_scale, kmeans_tsne_scale.labels_, metric='euclidean')))
        print(
            'KMeans tSNE Scaled davies bouldin Score: {}'.format(davies_bouldin_score(data, kmeans_tsne_scale.labels_)))

        labels_tsne_scale = kmeans_tsne_scale.labels_
        clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters': labels_tsne_scale})], axis=1)

        # plt.figure(figsize = (15,15))
        sns.scatterplot(clusters_tsne_scale.iloc[:, 0], clusters_tsne_scale.iloc[:, 1], hue=labels_tsne_scale,
                        palette=pallet, s=50, alpha=0.75).set_title(_title)  # , fontsize=15)
        plt.legend()
        plt.show()


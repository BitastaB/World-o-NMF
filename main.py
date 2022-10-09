import sys

from Models.DGONMF import sDGONMF, kDGONMF
from Models.ERWNMF import ERWNMF
from Models.GNMF import GNMF
from Models.DGRSNMF import GRDeepSNMF
from Models.GRSNMF import GRSNMF
from Models.NMF import NMF
from Models.OGNMF import OGNMF
from Models.RSCNMF import RSCNMF
from Models.dnsNMF import dnsNMF
import scipy.io

from Models.dsNMF import dsnmf
from Models.nsNMF import nsNMF
from Utils.utils import print_static


def load_dataset(dataset):
    if dataset == "cstr":
        imgData = scipy.io.loadmat('./Image_Data/CSTR.mat')
        matImg = imgData['fea'].astype('float32')
        matGnd = imgData['gnd']
        y = matGnd.ravel()

    if dataset == "jaffe":
        imgData = scipy.io.loadmat('./Image_Data/jaffe.mat')
        matImg = imgData['fea'].astype('float32')
        matGnd = imgData['gnd']
        y = matGnd.ravel()

    if dataset == "orl":
        imgData = scipy.io.loadmat('./Image_Data/ORL.mat')
        matImg = imgData['X'].astype('float32')
        matGnd = imgData['Y']
        y = matGnd.ravel()

    if dataset == "warpAR10P":
        imgData = scipy.io.loadmat('./Image_Data/warpAR10P.mat')
        matImg = imgData['X'].astype('float32')
        matGnd = imgData['Y']
        y = matGnd.ravel()

    if dataset == "umist":
        imgData = scipy.io.loadmat('./Image_Data/Umist.mat')
        matImg = imgData['X'].astype('float32')
        matGnd = imgData['Y']
        y = matGnd.ravel()

    if dataset == "yale":
        imgData = scipy.io.loadmat('./Image_Data/Yale_32x32.mat')
        matImg = imgData['fea'].astype('float32')
        matGnd = imgData['gnd']
        y = matGnd.ravel()

    if dataset == "yaleB":
        # imgData = scipy.io.loadmat('/home/bitasta/Desktop/NMF_Stuff/Image_Data/Yale_32x32.mat')
        imgData = scipy.io.loadmat('./Image_Data/YaleB_32x32.mat')
        matImg = imgData['fea'].astype('float32')
        matGnd = imgData['gnd']
        y = matGnd.ravel()

    if dataset == "coil20":
        imgData = scipy.io.loadmat('./Image_Data/COIL20.mat')
        matImg = imgData['X'].astype('float32')
        matGnd = imgData['y']
        y = matGnd.ravel()


    return imgData, matImg, matGnd, y


def run_model(model, dataset, alphas, betas, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, maxiter, eps_1, eps_2, y,
              maxiter_inner, pos_alpha_range, pos_beta_range, lambda_range, k_knn_list, plot_graphs):
    if model == "sDGONMF":
        sDGONMF.run_model(model, dataset, alphas, betas, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, maxiter, eps_1,
                            eps_2, y)
    elif model == "kDGONMF":
        kDGONMF.run_model(model, dataset, alphas, betas, matImg, matGnd, k1_list, k2_list, k_knn_list, maxiter_kmeans, l, maxiter, eps_1,
                            eps_2, y)

    elif model == "dnsNMF":
        dnsNMF.run_model(model, dataset, l, alphas, matImg, y, k1_list, k2_list, maxiter, maxiter_inner, maxiter_kmeans)

    elif model == "dsnmf":
        dsnmf.run_model(model, dataset, matImg, y, k1_list, k2_list, maxiter_kmeans)

    elif model == "ERWNMF":
        ERWNMF.run_model(model, dataset, matImg, y, k2_list, maxiter, maxiter_kmeans, plot_graphs)

    elif model == "RSCNMF":
        RSCNMF.run_model(model, dataset, matImg, y, pos_alpha_range, pos_beta_range, k_knn_list, k2_list, lambda_range,
                         maxiter_kmeans, maxiter)

    elif model == "OGNMF":
        OGNMF.run_model(model, dataset, matImg, y, alphas, betas, k_knn_list, k2_list, maxiter_kmeans, eps_1, eps_2, max_iter)

    elif model == "GRSNMF":
        GRSNMF.run_model(model, dataset, matImg, y, k_knn_list, k2_list, alpha_range, maxiter, maxiter_kmeans)

    elif model == "GNMF":
        GNMF.run_model(model, dataset, matImg, y, k_knn_list, k2_list, alpha_range, maxiter, maxiter_kmeans)

    elif model == "NMF":
        NMF.run_model(model, dataset, matImg, y, k2_list, maxiter_kmeans)

    elif model == "nsNMF":
        nsNMF.run_model(model, dataset, alphas, matImg, y, k2_list, maxiter, maxiter_inner, maxiter_kmeans)

    elif model == "DGRSNMF":
        GRDeepSNMF.run_model(model, dataset, matImg, y, k_knn_list, k1_list, k2_list, alphas, l, maxiter_kmeans, maxiter, maxiter_inner)


if __name__ == '__main__':
    dataset = "orl"
    model = "RSCNMF"  # Options : sDGONMF, kDGONMF, dnsNMF, dsnmf, ERWNMF, RSCNMF, OGNMF, GRSNMF, GNMF, NMF, nsNMF, DGRSNMF
    write_to_file = False
    plot_graphs = False

    if write_to_file:
        path = f"Results/{dataset}/output_new_{model}_{dataset}.out"
   #     path = f"Results/{dataset}/recon_{model}_{dataset}.out"
        sys.stdout = open(path, 'w')

    # Load dataset
    imgData, matImg, matGnd, y = load_dataset(dataset)

    # Setting parameters and hyper-parameters
    l = 2  # The number of layers
    k1_list = [80]# [80, 100, 120, 200]  # The size of the first layer
    k2_list = [10]#[10, 20, 30, 40, 50, 60, 70]
    alpha_range = [10] #[1e-03, 1e-02, 1e-01, 1e01]
    beta_range = [0.1]#[1e-02, 1e-01, 1]
    pos_alpha_range = [1e03]#[1e03, 1e04, 1e05, 1e06]
    pos_beta_range = [10] #[10, 100, 1000]
    lambda_range = [1, 10, 100]
    k_knn_list = [3, 5]#[3, 5, 6, 11, 21]
    max_iter = 100  # Maximum Number of Iterations
    maxiter_inner = 100
    maxiter_kmeans = 20
    eps_1 = 1e-12
    eps_2 = 1e-10

    # Print static params
    print_static(model, dataset, max_iter, eps_1, eps_2)

    # Run model
    run_model(model, dataset, alpha_range, beta_range, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, max_iter, eps_1,
              eps_2, y, maxiter_inner, pos_alpha_range, pos_beta_range, lambda_range, k_knn_list, plot_graphs)

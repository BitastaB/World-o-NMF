import sys

from Models.DGONMF import deepgonmf
from Models.ERWNMF import ERWNMF
from Models.GNMF import GNMF
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
        dataset = "CSTR.mat"

    if dataset == "jaffe":
        imgData = scipy.io.loadmat('./Image_Data/jaffe.mat')
        matImg = imgData['fea'].astype('float32')
        matGnd = imgData['gnd']
        y = matGnd.ravel()
        dataset = "jaffe.mat"

    if dataset == "orl":
        imgData = scipy.io.loadmat('./Image_Data/ORL.mat')
        matImg = imgData['X'].astype('float32')
        matGnd = imgData['Y']
        y = matGnd.ravel()
        dataset = "ORL.mat"

    if dataset == "warpAR10P":
        imgData = scipy.io.loadmat('./Image_Data/warpAR10P.mat')
        matImg = imgData['X'].astype('float32')
        matGnd = imgData['Y']
        y = matGnd.ravel()
        dataset = "warpAR10P.mat"

    if dataset == "umist":
        imgData = scipy.io.loadmat('./Image_Data/Umist.mat')
        matImg = imgData['X'].astype('float32')
        matGnd = imgData['Y']
        y = matGnd.ravel()
        dataset = "Umist.mat"

    if dataset == "yale":
        imgData = scipy.io.loadmat('./Image_Data/Yale_32x32.mat')
        matImg = imgData['fea'].astype('float32')
        matGnd = imgData['gnd']
        y = matGnd.ravel()
        dataset = "Yale_32x32.mat"

    if dataset == "yaleB":
        # imgData = scipy.io.loadmat('/home/bitasta/Desktop/NMF_Stuff/Image_Data/Yale_32x32.mat')
        imgData = scipy.io.loadmat('./Image_Data/YaleB_32x32.mat')
        matImg = imgData['fea'].astype('float32')
        matGnd = imgData['gnd']
        y = matGnd.ravel()
        dataset = "YaleB_32x32.mat"

    return imgData, matImg, matGnd, y, dataset


def run_model(model, alphas, betas, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, maxiter, eps_1, eps_2, y,
              maxiter_inner, pos_alpha_range, pos_beta_range, lambda_range, k_knn_list):
    if model == "DGONMF":
        deepgonmf.run_model(alphas, betas, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, maxiter, eps_1,
                            eps_2, y)

    elif model == "dnsNMF":
        dnsNMF.run_model(l, alphas, matImg, y, k1_list, k2_list, maxiter, maxiter_inner, maxiter_kmeans)

    elif model == "dsnmf":
        dsnmf.run_model(matImg, y, k1_list, k2_list, maxiter_kmeans)

    elif model == "ERWNMF":
        ERWNMF.run_model(matImg, y, k2_list, maxiter, maxiter_kmeans)

    elif model == "RSCNMF":
        RSCNMF.run_model(matImg, y, pos_alpha_range, pos_beta_range, k_knn_list, k2_list, lambda_range,
                         maxiter_kmeans, maxiter)

    elif model == "OGNMF":
        OGNMF.run_model(matImg, y, alphas, betas, k_knn_list, k2_list, maxiter_kmeans, eps_1, eps_2, max_iter)

    elif model == "GRSNMF":
        GRSNMF.run_model(matImg, y, k_knn_list, k2_list, alpha_range, maxiter, maxiter_kmeans)

    elif model == "GNMF":
        GNMF.run_model(matImg, y, k_knn_list, k2_list, alpha_range, maxiter, maxiter_kmeans)

    elif model == "NMF":
        NMF.run_model(matImg, y, k2_list, maxiter_kmeans)

    elif model == "nsNMF":
        nsNMF.run_model(alphas, matImg, y, k2_list, maxiter, maxiter_inner, maxiter_kmeans)


if __name__ == '__main__':
    dataset = "jaffe"
    model = "GNMF"  # Options : DGONMF, dnsNMF, dsnmf, ERWNMF, RSCNMF, OGNMF, GRSNMF, GNMF, NMF, nsNMF
    write_to_file = True

    if write_to_file:
        path = f"Results/output_{model}.out"
        sys.stdout = open(path, 'w')

    # Load dataset
    imgData, matImg, matGnd, y, dataset = load_dataset(dataset)

    # Setting parameters and hyper-parameters
    l = 2  # The number of layers
    k1_list = [80, 100, 120, 200]  # The size of the first layer
    k2_list = [10, 20, 30, 40, 50, 60, 70]
    alpha_range = [1e-03, 1e-02, 1e-01, 1e01]
    beta_range = [1e-02, 1e-01, 1]
    pos_alpha_range = [1e03, 1e04, 1e05, 1e06]
    pos_beta_range = [10, 100, 1000]
    lambda_range = [1, 10, 100]
    k_knn_list = [5]#[3, 5, 6, 11, 21]
    max_iter = 100  # Maximum Number of Iterations
    maxiter_inner = 100
    maxiter_kmeans = 20
    eps_1 = 1e-12
    eps_2 = 1e-10

    # Print static params
    print_static(model, dataset, max_iter, eps_1, eps_2)

    # Run model
    run_model(model, alpha_range, beta_range, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, max_iter, eps_1,
              eps_2, y, maxiter_inner, pos_alpha_range, pos_beta_range, lambda_range, k_knn_list)

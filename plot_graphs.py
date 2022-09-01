import numpy as np
from matplotlib import pyplot as plt

from Utils.plot_metrics import plot_clusters_tsne


def nsNMF_lists(dataset, rnd):

    path = f"Results/{dataset}/output_nsNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = []
        nmi_list = []

        for line in line_list:
            if "avg_acc" in line:
                acc_val = round(float((line.split("avg_acc = ")[1]).split(" , with ")[0]), rnd)
                acc_list.append(acc_val)

            if "avg_nmi" in line:
                nmi_val = round(float((line.split("avg_nmi = ")[1]).split(" , with ")[0]), rnd)
                nmi_list.append(nmi_val)

    return acc_list, nmi_list


def RSCNMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_RSCNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        for line in line_list:
            if "Avg ACC" in line:
                n = line_list.index(line)
                k = int((line_list[n - 1].split("k1 = ")[1]).split(" : alpha,")[0])
                acc_val = round(float((line.split("Avg ACC : ")[1]).split(", with ")[0]), rnd)
                nmi_val = round(float((line_list[n + 1].split("Avg NMI : ")[1]).split(", with ")[0]), rnd)

                if k not in acc_list.keys():
                    acc_list[k] = acc_val
                else:
                    if acc_val > acc_list[k]:
                        acc_list[k] = acc_val

                if k not in nmi_list.keys():
                    nmi_list[k] = nmi_val
                else:
                    if nmi_val > nmi_list[k]:
                        nmi_list[k] = nmi_val

    return list(acc_list.values()), list(nmi_list.values())


def GRSNFM_lists(dataset, rnd):
    path = f"Results/{dataset}/output_GRSNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        for line in line_list:
            if "Avg ACC" in line:
                n = line_list.index(line)
                k = int((line_list[n - 1].split("k = ")[1]).split(" :  ")[0])
                acc_val = round(float((line.split("Avg ACC : ")[1]).split(", with ")[0]), rnd)
                nmi_val = round(float((line_list[n + 1].split("Avg NMI : ")[1]).split(", with ")[0]), rnd)

                if k not in acc_list.keys():
                    acc_list[k] = acc_val
                else:
                    if acc_val > acc_list[k]:
                        acc_list[k] = acc_val

                if k not in nmi_list.keys():
                    nmi_list[k] = nmi_val
                else:
                    if nmi_val > nmi_list[k]:
                        nmi_list[k] = nmi_val

    return list(acc_list.values()), list(nmi_list.values())


def OGNMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_OGNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        for line in line_list:
            if "avg_acc" in line:
                n = line_list.index(line)
                k = int((line.split("k = ")[1]).split(" : best avg_acc")[0])
                acc_val = round(float((line.split("avg_acc = ")[1]).split(" , with ")[0]), rnd)
                nmi_val = round(float((line_list[n + 1].split("avg_nmi = ")[1]).split(" , with ")[0]), rnd)

                if k not in acc_list.keys():
                    acc_list[k] = acc_val
                else:
                    if acc_val > acc_list[k]:
                        acc_list[k] = acc_val

                if k not in nmi_list.keys():
                    nmi_list[k] = nmi_val
                else:
                    if nmi_val > nmi_list[k]:
                        nmi_list[k] = nmi_val

    return list(acc_list.values()), list(nmi_list.values())


def dsnmf_lists(dataset, rnd):
    path = f"Results/{dataset}/output_dsnmf_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        for line in line_list:
            if "avg_acc" in line:
                n = line_list.index(line)
                k = int((line.split("k2 = ")[1]).split(" : avg_acc")[0])
                acc_val = round(float(line.split("avg_acc = ")[1]), rnd)
                nmi_val = round(float(line_list[n + 1].split("avg_nmi = ")[1]), rnd)

                if k not in acc_list.keys():
                    acc_list[k] = acc_val
                else:
                    if acc_val > acc_list[k]:
                        acc_list[k] = acc_val

                if k not in nmi_list.keys():
                    nmi_list[k] = nmi_val
                else:
                    if nmi_val > nmi_list[k]:
                        nmi_list[k] = nmi_val

    return list(acc_list.values()), list(nmi_list.values())


def dnsNMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_dnsNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        for line in line_list:
            if "Max ACC" in line:
                n = line_list.index(line)
                k = int(line_list[n - 1].split("k2 = ")[1])
                acc_val = round((float((line.split("Max ACC : ")[1]).split(", with ")[0])), rnd)
                nmi_val = round((float((line_list[n + 1].split("Max NMI : ")[1]).split(", with ")[0])), rnd)
                if k not in acc_list.keys():
                    acc_list[k] = acc_val
                else:
                    if acc_val > acc_list[k]:
                        acc_list[k] = acc_val

                if k not in nmi_list.keys():
                    nmi_list[k] = nmi_val
                else:
                    if nmi_val > nmi_list[k]:
                        nmi_list[k] = nmi_val

    return list(acc_list.values()), list(nmi_list.values())


def DGONMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_DGONMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        for line in line_list:
            if "Avg ACC" in line:
                n = line_list.index(line)
                k = int(line_list[n - 1].split("k2 = ")[1])
                acc_val = round((float((line.split(" Avg ACC : ")[1]).split(", with ")[0])), rnd)
                nmi_val = round((float((line_list[n + 1].split(" Avg NMI : ")[1]).split(", with ")[0])), rnd)
                if k not in acc_list.keys():
                    acc_list[k] = acc_val
                else:
                    if acc_val > acc_list[k]:
                        acc_list[k] = acc_val

                if k not in nmi_list.keys():
                    nmi_list[k] = nmi_val
                else:
                    if nmi_val > nmi_list[k]:
                        nmi_list[k] = nmi_val

    return list(acc_list.values()), list(nmi_list.values())


# ERWNMF
def erwnmf_lists(dataset, rnd):
    path = f"Results/{dataset}/output_ERWNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = []
        nmi_list = []

        for line in line_list:
            if "avg_acc" in line:
                acc_val = round(float((line.split("best avg_acc = ")[1]).split(" , with nu")[0]), rnd)
                acc_list.append(acc_val)

            if "avg_nmi" in line:
                nmi_val = round(float((line.split("best avg_nmi = ")[1]).split(" , with nu")[0]), rnd)
                nmi_list.append(nmi_val)

    return acc_list, nmi_list


# Standard NMF
def nmf_lists(dataset, rnd):
    path = f"Results/{dataset}/output_NMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = []
        nmi_list = []

        for line in line_list:
            if "meanACC" in line:
                acc_val = round(float(line.split("meanACC = ")[1]), rnd)
                acc_list.append(acc_val)

            if "meanNMI" in line:
                nmi_val = round(float(line.split("meanNMI = ")[1]), rnd)
                nmi_list.append(nmi_val)

    return acc_list, nmi_list


# GNMF
def gnmf_lists(dataset, rnd):
    path = f"Results/{dataset}/output_GNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        for line in line_list:
            if "Avg ACC" in line:
                k = int((line.split(" :  Avg ACC")[0]).split("k = ")[1])
                acc_val = round(float((line.split("Avg ACC : ")[1]).split(", with lambda")[0]), rnd)
                if k not in acc_list.keys():
                    acc_list[k] = acc_val
                else:
                    if acc_val > acc_list[k]:
                        acc_list[k] = acc_val

            if "Avg NMI" in line:
                k = int((line.split(" :  Avg NMI")[0]).split("k = ")[1])
                nmi_val = round(float((line.split("Avg NMI : ")[1]).split(", with lambda")[0]), rnd)
                if k not in nmi_list.keys():
                    nmi_list[k] = nmi_val
                else:
                    if nmi_val > nmi_list[k]:
                        nmi_list[k] = nmi_val

        return list(acc_list.values()), list(nmi_list.values())


def plot_acc_nmi(dataset):
    round_upto = 4
    k_list = [10, 20, 30, 40, 50, 60, 70]

    # Get average accuracy and nmi values for models
    dgonmf_acc_list, dgonmf_nmi_list = DGONMF_lists(dataset, round_upto)
    erwnmf_acc_list, erwnmf_nmi_list = erwnmf_lists(dataset, round_upto)
    nmf_acc_list, nmf_nmi_list = nmf_lists(dataset, round_upto)
    gnmf_acc_list, gnmf_nmi_list = gnmf_lists(dataset, round_upto)
    dnsNMF_acc_list, dnsNMF_nmi_list = dnsNMF_lists(dataset, round_upto)
    dsNMF_acc_list, dsNMF_nmi_list = dsnmf_lists(dataset, round_upto)
    ognmf_acc_list, ognmf_nmi_list = OGNMF_lists(dataset, round_upto)
    grsnmf_acc_list, grsnmf_nmi_list = GRSNFM_lists(dataset, round_upto)
    rscnmf_acc_list, rscnmf_nmi_list = RSCNMF_lists(dataset, round_upto)
    nsnmf_acc_list, nsnmf_nmi_list = nsNMF_lists(dataset, round_upto)


    # Plotting average accuracy
    plt.plot(k_list, dgonmf_acc_list, label="DGONMF", marker='.')
    plt.plot(k_list, erwnmf_acc_list, label="ERWNMF", marker='.')
    plt.plot(k_list, nmf_acc_list, label="NMF", marker='.')
    plt.plot(k_list, gnmf_acc_list, label="GNMF", marker='.')
    plt.plot(k_list, dnsNMF_acc_list, label="dnsNMF", marker='.')
    plt.plot(k_list, dsNMF_acc_list, label="dsnmf", marker='.')
    plt.plot(k_list, ognmf_acc_list, label="OGNMF", marker='.')
    plt.plot(k_list, grsnmf_acc_list, label="GRSNMF", marker='.')
    plt.plot(k_list, rscnmf_acc_list, label="RSCNMF", marker='.')
    plt.plot(k_list, nsnmf_acc_list, label="nsNMF", marker='.')

    plt.xlabel("k")
    plt.ylabel("accurancy in %")
    plt.legend(loc="right")
    plt.title(f"Average accuracy for : {dataset}")
    plt.savefig(f'Results/plots/{dataset}/{dataset}_acc.png', dpi=150)
    plt.show()

    # Plotting average nmi
    plt.plot(k_list, dgonmf_nmi_list, label="DGONMF", marker='*')
    plt.plot(k_list, erwnmf_nmi_list, label="ERWNMF", marker='*')
    plt.plot(k_list, nmf_nmi_list, label="NMF", marker='*')
    plt.plot(k_list, gnmf_nmi_list, label="GNMF", marker='*')
    plt.plot(k_list, dnsNMF_nmi_list, label="dnsNMF", marker='*')
    plt.plot(k_list, dsNMF_nmi_list, label="dsnmf", marker='*')
    plt.plot(k_list, ognmf_nmi_list, label="OGNMF", marker='*')
    plt.plot(k_list, grsnmf_nmi_list, label="GRSNMF", marker='*')
    plt.plot(k_list, rscnmf_nmi_list, label="RSCNMF", marker='*')
    plt.plot(k_list, nsnmf_nmi_list, label="nsNMF", marker='*')

    plt.xlabel("k")
    plt.ylabel("nmi in %")
    plt.legend(loc="right")
    plt.title(f"Average nmi for : {dataset}")
    plt.savefig(f'Results/plots/{dataset}/{dataset}_nmi.png', dpi=150)
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


def plot_clusters():

    path = f'./Results/{dataset}/kmeans_{model}_{dataset}.npz'
    file = np.load(path)
    data = file['data']
    pred = file['kmneans_pred']
    n_cluster = np.unique(pred).shape[0]
    plot_clusters_tsne(data.T, model, dataset, kmeans_cluster=n_cluster)


if __name__ == '__main__':

    # jaffe, orl, warpAR10P, umist, yale, yaleB
    dataset = "jaffe"  # [jaffe, orl, warpAR10P, umist, yale, yaleB]
    model = "DGONMF"

    #plot_acc_nmi(dataset)
    #plot_model_performance("DGONMF")
    plot_clusters()

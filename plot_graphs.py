from matplotlib import pyplot as plt


## ERWNMF
def erwnmf_lists(rnd):
    with open('Results/jaffe/output_ERWNMF_jaffe.out') as f:
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
def nmf_lists(rnd):
    with open('Results/jaffe/output_NMF_jaffe.out') as f:
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
def gnmf_lists(rnd):
    with open('Results/jaffe/output_GNMF_jaffe.out') as f:
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


if __name__ == '__main__':
    round_upto = 4
    dataset = "jaffe"
    k_list = [10, 20, 30, 40, 50, 60, 70]

    # Get average accuracy and nmi values for models
    erwnmf_acc_list, erwnmf_nmi_list = erwnmf_lists(round_upto)
    nmf_acc_list, nmf_nmi_list = nmf_lists(round_upto)
    gnmf_acc_list, gnmf_nmi_list = gnmf_lists(round_upto)

    # Plotting average accuracy
    plt.plot(k_list, erwnmf_acc_list, label="ERWNMF", marker='o')
    plt.plot(k_list, nmf_acc_list, label="NMF", marker='v')
    plt.plot(k_list, gnmf_acc_list, label="GNMF", marker='+')

    plt.xlabel("k")
    plt.ylabel("accurancy in %")
    plt.legend()
    plt.title(f"Average accuracy for : {dataset}")
    plt.savefig(f'Results/plots/acc_{dataset}.png', dpi=150)
    plt.show()

    # Plotting average nmi
    plt.plot(k_list, erwnmf_nmi_list, label="ERWNMF", marker='o')
    plt.plot(k_list, nmf_nmi_list, label="NMF", marker='v')
    plt.plot(k_list, gnmf_nmi_list, label="GNMF", marker='+')

    plt.xlabel("k")
    plt.ylabel("nmi in %")
    plt.legend()
    plt.title(f"Average nmi for : {dataset}")
    plt.savefig(f'Results/plots/nmi_{dataset}.png', dpi=150)
    plt.show()

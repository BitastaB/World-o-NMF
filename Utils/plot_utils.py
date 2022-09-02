def check_key_dict(key, dict, val, com):
    if key not in dict.keys():
        dict[key] = val
    else:
        if com == 'gr':
            if val > dict[key]:
                dict[key] = val
        elif com == 'lr':
            if val < dict[key]:
                dict[key] = val
    return dict


def DGRSNMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_DGRSNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        sil_list = {}
        dunn_list = {}
        db_list = {}

        for line in line_list:
            if "Avg ACC : " in line:
                n = line_list.index(line)
                k2 = int((line_list[n - 1].split("k2 = ")[1]).split(" : k_knn")[0])
                acc_val = round(float((line.split("Avg ACC : ")[1]).split(", with ")[0]), rnd)
                nmi_val = round(float((line_list[n + 1].split("Avg NMI : ")[1]).split(", with ")[0]), rnd)
                sil_val = round(float((line_list[n + 3].split("Avg Silhoutte score : ")[1]).split(", with ")[0]), rnd)
                dunn_val = round(float((line_list[n + 4].split("Avg Dunn's Index score : ")[1]).split(", with ")[0]),
                                 rnd)
                db_val = round(float((line_list[n + 5].split("Bouldin score : ")[1]).split(", with ")[0]), rnd)

                acc_list = check_key_dict(k2, acc_list, acc_val, 'gr')
                nmi_list = check_key_dict(k2, nmi_list, nmi_val, 'gr')
                sil_list = check_key_dict(k2, sil_list, sil_val, 'gr')
                dunn_list = check_key_dict(k2, dunn_list, dunn_val, 'gr')
                db_list = check_key_dict(k2, db_list, db_val, 'lr')

    return list(acc_list.values()), list(nmi_list.values()), list(sil_list.values()), list(dunn_list.values()), list(
        db_list.values())


def nsNMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_nsNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = []
        nmi_list = []
        sil_list = []
        dunn_list = []
        db_list = []

        for line in line_list:
            if "avg acc" in line:
                acc_val = round(float((line.split("avg acc = ")[1]).split(" , with ")[0]), rnd)
                acc_list.append(acc_val)

            if "avg nmi" in line:
                nmi_val = round(float((line.split("avg nmi = ")[1]).split(" , with ")[0]), rnd)
                nmi_list.append(nmi_val)

            if "avg Silhoutte score" in line:
                sil_val = round(float((line.split("avg Silhoutte score = ")[1]).split(" , with ")[0]), rnd)
                sil_list.append(sil_val)

            if "avg Dunn's Index score" in line:
                dunn_val = round(float((line.split("avg Dunn's Index score = ")[1]).split(" , with ")[0]), rnd)
                dunn_list.append(dunn_val)

            if "avg Davies Bouldin score" in line:
                db_val = round(float((line.split("avg Davies Bouldin score = ")[1]).split(" , with ")[0]), rnd)
                db_list.append(db_val)

    return acc_list, nmi_list, sil_list, dunn_list, db_list


def RSCNMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_RSCNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        sil_list = {}
        dunn_list = {}
        db_list = {}

        for line in line_list:
            if "Avg ACC" in line:
                n = line_list.index(line)
                k = int((line_list[n - 1].split("k1 = ")[1]).split(" : alpha,")[0])
                acc_val = round(float((line.split("Avg ACC : ")[1]).split(", with ")[0]), rnd)
                nmi_val = round(float((line_list[n + 1].split("Avg NMI : ")[1]).split(", with ")[0]), rnd)
                sil_val = round(float((line_list[n + 2].split("Avg Silhoutte score : ")[1]).split(", with ")[0]), rnd)
                dunn_val = round(float((line_list[n + 3].split("Avg Dunn's Index score : ")[1]).split(", with ")[0]),
                                 rnd)
                db_val = round(float((line_list[n + 4].split("Avg Davies Bouldin score : ")[1]).split(", with ")[0]),
                               rnd)

                acc_list = check_key_dict(k, acc_list, acc_val, 'gr')
                nmi_list = check_key_dict(k, nmi_list, nmi_val, 'gr')
                sil_list = check_key_dict(k, sil_list, sil_val, 'gr')
                dunn_list = check_key_dict(k, dunn_list, dunn_val, 'gr')
                db_list = check_key_dict(k, db_list, db_val, 'lr')

    return list(acc_list.values()), list(nmi_list.values()), list(sil_list.values()), list(dunn_list.values()), list(
        db_list.values())


def GRSNFM_lists(dataset, rnd):
    path = f"Results/{dataset}/output_GRSNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        sil_list = {}
        dunn_list = {}
        db_list = {}

        for line in line_list:
            if "Avg ACC" in line:
                n = line_list.index(line)
                k = int((line_list[n - 1].split("k = ")[1]).split(" :  ")[0])
                acc_val = round(float((line.split("Avg ACC : ")[1]).split(", with ")[0]), rnd)
                nmi_val = round(float((line_list[n + 1].split("Avg NMI : ")[1]).split(", with ")[0]), rnd)
                sil_val = round(float((line_list[n + 3].split("Avg Silhoutte score : ")[1]).split(", with ")[0]), rnd)
                dunn_val = round(float((line_list[n + 4].split("Avg Dunn's Index score : ")[1]).split(", with ")[0]),
                                 rnd)
                db_val = round(float((line_list[n + 5].split("Avg David Bouldin score : ")[1]).split(", with ")[0]),
                               rnd)

                acc_list = check_key_dict(k, acc_list, acc_val, 'gr')
                nmi_list = check_key_dict(k, nmi_list, nmi_val, 'gr')
                sil_list = check_key_dict(k, sil_list, sil_val, 'gr')
                dunn_list = check_key_dict(k, dunn_list, dunn_val, 'gr')
                db_list = check_key_dict(k, db_list, db_val, 'lr')

    return list(acc_list.values()), list(nmi_list.values()), list(sil_list.values()), list(dunn_list.values()), list(
        db_list.values())


def OGNMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_OGNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        sil_list = {}
        dunn_list = {}
        db_list = {}

        for line in line_list:
            if "avg acc" in line:
                n = line_list.index(line)
                k = int((line.split("k = ")[1]).split(" : best avg acc")[0])
                acc_val = round(float((line.split("avg acc = ")[1]).split(" , with ")[0]), rnd)
                nmi_val = round(float((line_list[n + 1].split("avg nmi = ")[1]).split(" , with ")[0]), rnd)
                sil_val = round(float((line_list[n + 2].split("avg Silhoutte score = ")[1]).split(" , with ")[0]), rnd)
                dunn_val = round(float((line_list[n + 3].split("avg Dunn's Index score = ")[1]).split(" , with ")[0]),
                                 rnd)
                db_val = round(float((line_list[n + 4].split("avg Davies Bouldin score = ")[1]).split(" , with ")[0]),
                               rnd)

                acc_list = check_key_dict(k, acc_list, acc_val, 'gr')
                nmi_list = check_key_dict(k, nmi_list, nmi_val, 'gr')
                sil_list = check_key_dict(k, sil_list, sil_val, 'gr')
                dunn_list = check_key_dict(k, dunn_list, dunn_val, 'gr')
                db_list = check_key_dict(k, db_list, db_val, 'lr')

    return list(acc_list.values()), list(nmi_list.values()), list(sil_list.values()), list(dunn_list.values()), list(
        db_list.values())


def dsnmf_lists(dataset, rnd):
    path = f"Results/{dataset}/output_dsnmf_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        sil_list = {}
        dunn_list = {}
        db_list = {}

        for line in line_list:
            if "avg_acc" in line:
                n = line_list.index(line)
                k = int((line.split("k2 = ")[1]).split(" : avg_acc")[0])
                acc_val = round(float(line.split("avg_acc = ")[1]), rnd)
                nmi_val = round(float(line_list[n + 1].split("avg_nmi = ")[1]), rnd)
                sil_val = round(float(line_list[n + 2].split("avg_Silhoutte score = ")[1]), rnd)
                dunn_val = round(float(line_list[n + 3].split("avg_Dunn'd index score = ")[1]), rnd)
                db_val = round(float(line_list[n + 4].split("avg_Davies bouldin score = ")[1]), rnd)

                acc_list = check_key_dict(k, acc_list, acc_val, 'gr')
                nmi_list = check_key_dict(k, nmi_list, nmi_val, 'gr')
                sil_list = check_key_dict(k, sil_list, sil_val, 'gr')
                dunn_list = check_key_dict(k, dunn_list, dunn_val, 'gr')
                db_list = check_key_dict(k, db_list, db_val, 'lr')

    return list(acc_list.values()), list(nmi_list.values()), list(sil_list.values()), list(
        dunn_list.values()), list(
        db_list.values())


def dnsNMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_dnsNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        sil_list = {}
        dunn_list = {}
        db_list = {}

        for line in line_list:
            if "Max ACC" in line:
                n = line_list.index(line)
                k = int(line_list[n - 1].split("k2 = ")[1])
                acc_val = round((float((line.split("Max ACC : ")[1]).split(", with ")[0])), rnd)
                nmi_val = round((float((line_list[n + 1].split("Max NMI : ")[1]).split(", with ")[0])), rnd)
                sil_val = round((float((line_list[n + 3].split("Max Silhoutter score : ")[1]).split(", with ")[0])),
                                rnd)
                dunn_val = round((float((line_list[n + 4].split("Max Dunn's Index score : ")[1]).split(", with ")[0])),
                                 rnd)
                db_val = round((float((line_list[n + 5].split("Min David Bouldin score : ")[1]).split(", with ")[0])),
                               rnd)

                acc_list = check_key_dict(k, acc_list, acc_val, 'gr')
                nmi_list = check_key_dict(k, nmi_list, nmi_val, 'gr')
                sil_list = check_key_dict(k, sil_list, sil_val, 'gr')
                dunn_list = check_key_dict(k, dunn_list, dunn_val, 'gr')
                db_list = check_key_dict(k, db_list, db_val, 'lr')

    return list(acc_list.values()), list(nmi_list.values()), list(sil_list.values()), list(
        dunn_list.values()), list(
        db_list.values())


def DGONMF_lists(dataset, rnd):
    path = f"Results/{dataset}/output_DGONMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        sil_list = {}
        dunn_list = {}
        db_list = {}

        for line in line_list:
            if "Avg ACC" in line:
                n = line_list.index(line)
                k = int(line_list[n - 1].split("k2 = ")[1])
                acc_val = round((float((line.split(" Avg ACC : ")[1]).split(", with ")[0])), rnd)
                nmi_val = round((float((line_list[n + 1].split(" Avg NMI : ")[1]).split(", with ")[0])), rnd)
                sil_val = round((float((line_list[n + 3].split(" Avg Silhoutte score : ")[1]).split(", with ")[0])),
                                rnd)
                dunn_val = round((float((line_list[n + 4].split(" Avg Dunn's Index score : ")[1]).split(", with ")[0])),
                                 rnd)
                db_val = round((float((line_list[n + 5].split(" Avg David Bouldin score : ")[1]).split(", with ")[0])),
                               rnd)

                acc_list = check_key_dict(k, acc_list, acc_val, 'gr')
                nmi_list = check_key_dict(k, nmi_list, nmi_val, 'gr')
                sil_list = check_key_dict(k, sil_list, sil_val, 'gr')
                dunn_list = check_key_dict(k, dunn_list, dunn_val, 'gr')
                db_list = check_key_dict(k, db_list, db_val, 'lr')

    return list(acc_list.values()), list(nmi_list.values()), list(sil_list.values()), list(
        dunn_list.values()), list(
        db_list.values())


# ERWNMF
def erwnmf_lists(dataset, rnd):
    path = f"Results/{dataset}/output_ERWNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = []
        nmi_list = []
        sil_list = []
        dunn_list = []
        db_list = []

        for line in line_list:
            if "avg_acc" in line:
                acc_val = round(float((line.split("best avg_acc = ")[1]).split(" , with nu")[0]), rnd)
                acc_list.append(acc_val)

            if "avg_nmi" in line:
                nmi_val = round(float((line.split("best avg_nmi = ")[1]).split(" , with nu")[0]), rnd)
                nmi_list.append(nmi_val)

            if "avg_silhoutte score" in line:
                sil_val = round(float((line.split("best avg_silhoutte score = ")[1]).split(" , with nu")[0]), rnd)
                sil_list.append(sil_val)

            if "avg_dunn's index" in line:
                dunn_val = round(float((line.split("best avg_dunn's index score = ")[1]).split(" , with nu")[0]), rnd)
                dunn_list.append(dunn_val)

            if "lowest avg_davis bouldin score" in line:
                db_val = round(float((line.split("lowest avg_davis bouldin score = ")[1]).split(" , with nu")[0]), rnd)
                db_list.append(db_val)

    return acc_list, nmi_list, sil_list, dunn_list, db_list


# Standard NMF
def nmf_lists(dataset, rnd):
    path = f"Results/{dataset}/output_NMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = []
        nmi_list = []
        sil_list = []
        dunn_list = []
        db_list = []

        for line in line_list:
            if "mean ACC" in line:
                acc_val = round(float(line.split("mean ACC = ")[1]), rnd)
                acc_list.append(acc_val)

            if "mean NMI" in line:
                nmi_val = round(float(line.split("mean NMI = ")[1]), rnd)
                nmi_list.append(nmi_val)

            if "mean Silhoutte score" in line:
                sil_val = round(float(line.split("mean Silhoutte score = ")[1]), rnd)
                sil_list.append(sil_val)

            if "mean Dunn's Index Score" in line:
                dunn_val = round(float(line.split("mean Dunn's Index Score = ")[1]), rnd)
                dunn_list.append(dunn_val)

            if "mean Davies bouldin Score" in line:
                db_val = round(float(line.split("mean Davies bouldin Score = ")[1]), rnd)
                db_list.append(db_val)

    return acc_list, nmi_list, sil_list, dunn_list, db_list


# GNMF
def gnmf_lists(dataset, rnd):
    path = f"Results/{dataset}/output_GNMF_{dataset}.out"
    with open(path) as f:
        line_list = list(f)
        acc_list = {}
        nmi_list = {}
        sil_list = {}
        dunn_list = {}
        db_list = {}

        for line in line_list:
            if "Avg ACC" in line:
                n = line_list.index(line)
                k = int((line.split(" :  Avg ACC")[0]).split("k = ")[1])

                acc_val = round(float((line.split("Avg ACC : ")[1]).split(", with theta")[0]), rnd)
                nmi_val = round(float((line_list[n + 1].split("Avg NMI : ")[1]).split(", with theta")[0]), rnd)
                sil_val = round(float((line_list[n + 3].split("Avg Silhoutte score : ")[1]).split(", with theta")[0]),
                                rnd)
                dunn_val = round(
                    float((line_list[n + 4].split("Avg Dunn's Index score : ")[1]).split(", with theta")[0]), rnd)
                db_val = round(
                    float((line_list[n + 5].split("Avg David Bouldin score : ")[1]).split(", with theta")[0]), rnd)

                acc_list = check_key_dict(k, acc_list, acc_val, 'gr')
                nmi_list = check_key_dict(k, nmi_list, nmi_val, 'gr')
                sil_list = check_key_dict(k, sil_list, sil_val, 'gr')
                dunn_list = check_key_dict(k, dunn_list, dunn_val, 'gr')
                db_list = check_key_dict(k, db_list, db_val, 'lr')

    return list(acc_list.values()), list(nmi_list.values()), list(sil_list.values()), list(
        dunn_list.values()), list(
        db_list.values())

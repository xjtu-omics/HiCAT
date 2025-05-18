import Levenshtein
import pandas as pd
import numpy as np
import networkx as nx
import argparse
from joblib import Parallel, delayed
import datetime
import networkx.algorithms.community as nx_comm

import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.lines import Line2D

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'


def buildMonomerBlockSequence(decomposition_path):
    block_sequence = []
    with open(decomposition_path, 'r') as bf:
        while True:
            line = bf.readline()[:-1]
            if not line:
                break
            items = line.split('\t')
            if items[1][-1] == "'":
                block_sequence.append(items[0] + '_' + str(items[2]) + '_' + str(items[3]) + '_-')
            else:
                block_sequence.append(items[0] + '_' + str(items[2]) + '_' + str(items[3]) + '_+')

    return block_sequence


def ed_distance(sequenceA, sequenceB):
    return Levenshtein.distance(sequenceA, sequenceB) / max(len(sequenceA), len(sequenceB))


def ed_distance_apply(target, col):
    return col.apply(ed_distance, args=(target,))


def ed_distance_apply_apply(data, region):
    return data[region[0]:region[1]]['seq'].apply(ed_distance_apply, args=(data['seq'],))


def reverse(sequence):
    base_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    new_sequence = ''
    for i in sequence[::-1]:
        new_sequence += base_map[i]
    return new_sequence


def calculateED(block_sequence, base_sequence, thread):
    names = []
    seq = []
    for i in block_sequence:
        names.append(i)
        items = i.split('_')
        strand = items[3]
        if strand == '-':
            split_base_sequence = reverse(base_sequence[int(items[1]):int(items[2])])
        else:
            split_base_sequence = base_sequence[int(items[1]):int(items[2])]
        seq.append(split_base_sequence)

    data = pd.DataFrame({'name': names, 'seq': seq})
    binsize = int(len(data) / thread) + 1
    split_in = [[binsize * i, binsize * i + binsize] for i in range(thread)]
    res = Parallel(n_jobs=thread)(delayed(ed_distance_apply_apply)(data, i) for i in split_in)
    res = pd.concat(res)
    edit_distance_matrix = np.array(res)
    block_name_index = list(data['name'])
    return edit_distance_matrix, block_name_index


def pre_Clustering(edit_distance_matrix, block_name_index):
    edit_distance_matrix = np.asarray(edit_distance_matrix)
    G = nx.Graph()
    for i in range(len(block_name_index)):
        for j in range(len(block_name_index)):
            if i == j:
                continue
            else:
                if edit_distance_matrix[i][j] == 0:
                    G.add_edge(i, j)
    G_components = list(nx.connected_components(G))
    pre_merge = {}
    marker_block = set()
    for i in G_components:
        component = list(i)
        pre_merge[component[0]] = []
        for j in component:
            pre_merge[component[0]].append(j)
            marker_block.add(block_name_index[j])

    final_pre_merge = {}
    for i in pre_merge.keys():
        final_pre_merge[block_name_index[i]] = []
        for j in pre_merge[i]:
            final_pre_merge[block_name_index[i]].append(block_name_index[j])
    for i in block_name_index:
        if i in marker_block:
            continue
        else:
            final_pre_merge[i] = [i]

    save_set = set(final_pre_merge.keys())
    merge_edit_distance_matrix = []
    merge_block_name_index = []
    for i in range(len(block_name_index)):
        if block_name_index[i] not in save_set:
            continue
        merge_block_name_index.append(block_name_index[i])
        distance_vector = []
        for j in range(len(block_name_index)):
            if block_name_index[j] not in save_set:
                continue
            distance_vector.append(edit_distance_matrix[i][j])
        merge_edit_distance_matrix.append(distance_vector)

    return merge_edit_distance_matrix, merge_block_name_index, final_pre_merge


def miningMonomerTDPattern(new_monomer_sequence, max_hor_len):
    new_monomer_sequence_index_left = []
    new_monomer_sequence_index_right = []
    for i in range(len(new_monomer_sequence)):
        new_monomer_sequence_index_left.append(i)
        new_monomer_sequence_index_right.append(i)
    top_layer = []
    all_layer = []
    start_d = 1

    while start_d < min(max_hor_len, len(new_monomer_sequence)):
        monomer_sequence = new_monomer_sequence
        monomer_sequence_index_left = new_monomer_sequence_index_left
        monomer_sequence_index_right = new_monomer_sequence_index_right
        candidate_pattern = {}
        for i in range(len(monomer_sequence)):
            if str(monomer_sequence[i]) not in candidate_pattern.keys():
                candidate_pattern[str(monomer_sequence[i])] = [[i, i + 1, str(monomer_sequence[i])]]
            else:
                candidate_pattern[str(monomer_sequence[i])].append([i, i + 1, str(monomer_sequence[i])])

        d_database = []
        for i in candidate_pattern.keys():
            pattern_database = candidate_pattern[i]
            for j in range(len(pattern_database)):
                current = pattern_database[j]
                current_start = current[0]
                if j == 0:
                    continue
                candidate_database = pattern_database[:j]
                for k in candidate_database[::-1]:
                    pre_database = k
                    pre_start = pre_database[0]
                    d = current_start - pre_start
                    if d == start_d:
                        d_database.append([pre_database, current])
                    if d > start_d:
                        break

        if len(d_database) == 0:
            start_d += 1
            continue
        else:
            sorted_d_database = sorted(d_database, key=lambda x: x[0])
            chain_list = []
            chain = []
            chain.append(sorted_d_database[0][0])
            index = 1
            while index < len(sorted_d_database):
                if sorted_d_database[index][0][0] - sorted_d_database[index - 1][0][0] != 1:
                    final_start = index - start_d
                    if final_start >= 0:
                        for i in range(start_d):
                            chain.append(sorted_d_database[final_start + i][1])
                    if int(len(chain) / start_d) > 1:
                        chain_list.append(chain[:int(len(chain) / start_d) * start_d])
                    chain = [sorted_d_database[index][0]]
                    index += 1
                else:
                    chain.append(sorted_d_database[index][0])
                    index += 1
                    if index >= len(sorted_d_database):
                        final_start = index - start_d
                        if final_start >= 0:
                            for i in range(start_d):
                                chain.append(sorted_d_database[final_start + i][1])
                        if int(len(chain) / start_d) > 1:
                            chain_list.append(chain[:int(len(chain) / start_d) * start_d])
                        chain = []
                        break
                    while sorted_d_database[index][0][0] - sorted_d_database[index - 1][0][0] == 1:
                        chain.append(sorted_d_database[index][0])
                        index += 1
                        if index >= len(sorted_d_database):
                            break
                    final_start = index - start_d
                    if final_start >= 0:
                        for i in range(start_d):
                            chain.append(sorted_d_database[final_start + i][1])
                    if int(len(chain) / start_d) > 1:
                        chain_list.append(chain[:int(len(chain) / start_d) * start_d])
                    chain = []
            if len(chain) != 0:
                final_start = index - start_d
                if final_start >= 0:
                    for i in range(start_d):
                        chain.append(sorted_d_database[final_start + i][1])
                if int(len(chain) / start_d) > 1:
                    chain_list.append(chain[:int(len(chain) / start_d) * start_d])

            tmp_top_layer = []
            for i in range(len(chain_list)):
                relative_start = chain_list[i][0][0]
                relative_end = chain_list[i][-1][0]
                absolute_start = monomer_sequence_index_right[relative_start]

                TD_item = monomer_sequence[relative_start:relative_start + start_d]
                TD_count = int((relative_end - relative_start + 1) / start_d)
                TD_range_index = []
                relative_end = chain_list[i][int(len(chain_list[i]) / start_d) * start_d - 1][0]
                absolute_end = monomer_sequence_index_left[relative_end]
                new_start_td_item = relative_start
                new_end_td_item = relative_start + start_d - 1
                pattern_number = 1
                while True:
                    start_td_item = new_start_td_item
                    end_td_item = new_end_td_item
                    if end_td_item > relative_end:
                        break
                    TD_range_index.append(
                        [monomer_sequence_index_right[start_td_item], monomer_sequence_index_left[end_td_item],
                         TD_item, pattern_number])
                    pattern_number = 1
                    new_start_td_item = end_td_item + 1
                    new_end_td_item = end_td_item + start_d

                top_layer_ins = [absolute_start, absolute_end, TD_item, TD_count, TD_range_index, relative_start,
                                 relative_end]

                overlap_flag = 0
                for td in tmp_top_layer:
                    td_relative_start = td[5]
                    td_relative_end = td[6]
                    if (not (td_relative_end < relative_start or relative_end < td_relative_start)):
                        overlap_flag = 1
                        break

                if overlap_flag == 0:
                    tmp_top_layer.append(top_layer_ins)
                    all_layer.append(top_layer_ins)

            if len(tmp_top_layer) == 0:
                start_d += 1
                continue

            new_monomer_sequence_index_left = []
            new_monomer_sequence_index_right = []
            tmp_monomer_sequence_left = []
            tmp_monomer_sequence_right = []
            new_monomer_sequence = []
            for i in monomer_sequence:
                tmp_monomer_sequence_left.append(i)
                tmp_monomer_sequence_right.append(i)

            for i in tmp_top_layer:
                relative_start = i[5]
                relative_end = i[6]
                for j in range(relative_end - relative_start + 1):
                    tmp_monomer_sequence_left[relative_start + j] = '*'
                    tmp_monomer_sequence_right[relative_end - j] = '*'

                for j in range(start_d):
                    tmp_monomer_sequence_left[relative_start + j] = monomer_sequence[relative_start + j]
                    tmp_monomer_sequence_right[relative_end - j] = monomer_sequence[relative_end - j]

            for i in range(len(tmp_monomer_sequence_left)):
                if tmp_monomer_sequence_left[i] == '*':
                    continue
                else:
                    new_monomer_sequence.append(monomer_sequence[i])
                    new_monomer_sequence_index_left.append(monomer_sequence_index_left[i])

            for i in range(len(tmp_monomer_sequence_right)):
                if tmp_monomer_sequence_right[i] == '*':
                    continue
                else:
                    new_monomer_sequence_index_right.append(monomer_sequence_index_right[i])

            start_d = 1

    all_layer.sort(key=lambda x: (x[0], -x[1]))
    max_end = -float('inf')

    for unit in all_layer:
        if unit[1] <= max_end:
            continue
        else:
            max_end = unit[1]
            top_layer.append(unit)

    return new_monomer_sequence, top_layer, all_layer


def buildingHor(block_sequence, all_layer):
    final_HOR = {}
    for i in all_layer:
        start = i[0]
        end = i[1]
        start_block = block_sequence[start].split('_')
        strand = start_block[-1]
        pattern = i[2]
        pattern_range = i[4]
        in_flag = 0
        if strand == '-':
            pattern = pattern[::-1]

        for j in range(len(pattern)):  # 循环pattern
            prefix_pattern = pattern[j:]
            suffix_pattern = pattern[:j]
            loop_pattern = prefix_pattern + suffix_pattern
            s_loop_pattern = ''
            for k in loop_pattern:
                s_loop_pattern += str(k) + '_'
            s_loop_pattern = s_loop_pattern[:-1]
            if s_loop_pattern in final_HOR.keys():
                in_flag = 1
                final_HOR[s_loop_pattern][0].append([start, end])
                new_pattern_range = []
                for k in pattern_range:
                    new_pattern_range.append([k[0], k[1], strand, k[2], k[3]])
                final_HOR[s_loop_pattern][1] += new_pattern_range
                break

        if in_flag == 0:
            s_pattern = ''
            for j in pattern:
                s_pattern += str(j) + '_'
            s_pattern = s_pattern[:-1]
            new_pattern_range = []
            for j in pattern_range:
                new_pattern_range.append([j[0], j[1], strand, j[2], j[3]])
            final_HOR[s_pattern] = [[[start, end]], new_pattern_range]
    return final_HOR


# update rename monomer
def clusterAndFindingHOR(sk, min_similarity, step, merge_edit_distance_matrix, merge_block_name_index,
                         block_sequence, final_pre_merge, block_name_index, max_hor_len, log_file):
    # print('++++++' + str(similarity) + '++++++')
    # time_start = datetime.datetime.now()
    # build graph
    similarity = min_similarity + sk * step
    community_time_start = datetime.datetime.now()
    G = nx.Graph()
    for i in range(len(merge_block_name_index)):
        for j in range(len(merge_block_name_index)):
            if i == j:
                continue
            block_name_A = merge_block_name_index[i]
            block_name_B = merge_block_name_index[j]
            if (1 - merge_edit_distance_matrix[i][j]) >= similarity:
                G.add_edge(block_name_A, block_name_B)
    community = []
    if len(G.edges) != 0:
        community = nx_comm.louvain_communities(G, seed=1)
        for i in merge_block_name_index:
            if i not in G.nodes:
                community.append({i})
    else:
        for i in merge_block_name_index:
            community.append({i})
    community_time_end = datetime.datetime.now()
    log_file.write('community Time: ' + str((community_time_end - community_time_start).seconds) + '\n')
    log_file.flush()

    pre_mining_time_start = datetime.datetime.now()
    monomer_label = {}
    cluster_index = 1
    cluster_block = {}
    community_block_set = set()
    for i in community:
        cluster_block[cluster_index] = []
        for j in i:
            monomers = final_pre_merge[j]
            for k in monomers:
                monomer_label[k] = cluster_index
                cluster_block[cluster_index].append(k)
                community_block_set.add(k)
        cluster_index += 1
    for i in block_name_index:
        if i not in community_block_set:
            monomer_label[i] = cluster_index
            cluster_block[cluster_index] = [i]
            cluster_index += 1

    monomer_sequence = []
    for i in block_sequence:
        monomer_sequence.append(monomer_label[i])
    pre_mining_time_end = datetime.datetime.now()
    log_file.write('pre minning Time: ' + str((pre_mining_time_end - pre_mining_time_start).seconds) + '\n')
    log_file.flush()

    filter_HORs_list = {}
    mining_time_start = datetime.datetime.now()
    new_monomer_sequence, top_layer, all_layer = miningMonomerTDPattern(monomer_sequence, max_hor_len)
    mining_time_end = datetime.datetime.now()
    log_file.write('minning Time: ' + str((mining_time_end - mining_time_start).seconds) + '\n')
    log_file.flush()

    buildHOR_time_start = datetime.datetime.now()
    HORs = buildingHor(block_sequence, all_layer)
    # 是否需要过滤？？去掉
    filter_HORs = {}
    for i in HORs.keys():
        pattern = i
        database = HORs[i][1]
        filter_HOR = []
        init_sequences = []
        for j in monomer_sequence:
            init_sequences.append(0)
        sort_database = sorted(database, key=lambda x: x[1] - x[0])
        for j in sort_database:
            filter_HOR.append(j)
        if len(filter_HOR) == 0:
            continue
        else:
            filter_HORs[pattern] = sorted(filter_HOR, key=lambda x: x[0])

    for i in filter_HORs.keys():
        pattern = i.split('_')
        pattern_database = filter_HORs[i]
        in_flag = 0
        for j in range(len(pattern)):
            prefix_pattern = pattern[j:]
            suffix_pattern = pattern[:j]
            loop_pattern = prefix_pattern + suffix_pattern
            s_loop_pattern = ''
            for k in loop_pattern:
                s_loop_pattern += str(k) + '_'
            s_loop_pattern = s_loop_pattern[:-1]
            if s_loop_pattern in filter_HORs_list.keys():
                in_flag = 1
                for k in pattern_database:
                    filter_HORs_list[s_loop_pattern].append([k[0], k[1], k[2], k[3], k[4]])
                break
        if in_flag == 0:
            s_pattern = ''
            for j in pattern:
                s_pattern += str(j) + '_'
            s_pattern = s_pattern[:-1]
            filter_HORs_list[s_pattern] = []
            for j in pattern_database:
                filter_HORs_list[s_pattern].append([j[0], j[1], j[2], j[3], j[4]])

    pattern_score = []
    for i in filter_HORs_list.keys():
        cov = 0
        for j in filter_HORs_list[i]:
            cov += (j[1] + 1 - j[0])
        cov_rate = cov / len(monomer_sequence)
        pattern = i.split('_')
        repeat_number = 0
        for j in filter_HORs_list[i]:
            repeat_number += j[4]
        pattern_rate = repeat_number / (cov / len(pattern))
        score = cov_rate * pattern_rate
        pattern_score.append([score, cov_rate, pattern_rate, i])

    pattern_score = sorted(pattern_score, key=lambda x: x[0], reverse=True)
    filter_final_HORs = {}
    marker_sequence = []
    for i in monomer_sequence:
        marker_sequence.append(0)
    for i in pattern_score:
        database = filter_HORs_list[i[3]]
        for j in database:
            start = j[0]
            end = j[1]
            for k in range(start, end + 1):
                marker_sequence[k] = 1
        filter_final_HORs[i[3]] = database

    cov = 0
    for i in marker_sequence:
        if i == 1:
            cov += 1

    # rename monomer index based on R1, and remain based on frequence
    # rename
    # 1 cluster_block
    # 2 monomer_sequence
    # 3 filter_final_HORs
    # 4 top_layer
    # 5 all_layer

    rename_monomer_ID = {}
    # getFirst HOR pattern
    all_patterns = list(filter_final_HORs.keys())
    if len(all_patterns) != 0:
        first_pattern_rename = {}
        first_pattern = all_patterns[0].split('_')
        monomer_ID = 1
        for i in first_pattern:
            if int(i) not in first_pattern_rename.keys():
                first_pattern_rename[int(i)] = monomer_ID
                monomer_ID += 1
        monomer_out_of_first_pattern = {}
        sorted_cluster_block = sorted(cluster_block.items(), key=lambda x: len(x[1]), reverse=True)
        for i in sorted_cluster_block:
            if i[0] not in first_pattern_rename.keys():
                if i[0] not in monomer_out_of_first_pattern.keys():
                    monomer_out_of_first_pattern[i[0]] = monomer_ID
                    monomer_ID += 1
        for i in first_pattern_rename.keys():
            rename_monomer_ID[i] = first_pattern_rename[i]
        for i in monomer_out_of_first_pattern.keys():
            rename_monomer_ID[i] = monomer_out_of_first_pattern[i]

    else:
        monomer_ID = 1
        sorted_cluster_block = sorted(cluster_block.items(), key=lambda x: len(x[1]), reverse=True)
        for i in sorted_cluster_block:
            rename_monomer_ID[i[0]] = monomer_ID
            monomer_ID += 1
    # using rename_monomer_ID update
    # 1 cluster_block
    # 2 monomer_sequence
    # 3 filter_final_HORs
    # 4 top_layer
    # 5 all_layer

    update_cluster_block = {}
    update_monomer_sequence = []
    update_filter_final_HORs = {}
    update_top_layer = []
    update_all_layer = []

    for i in cluster_block.keys():
        update_cluster_block[rename_monomer_ID[i]] = cluster_block[i]

    for i in monomer_sequence:
        update_monomer_sequence.append(rename_monomer_ID[i])

    for i in filter_final_HORs.keys():
        pattern = i.split('_')
        update_pattern = ''
        for j in pattern:
            update_pattern += str(rename_monomer_ID[int(j)]) + '_'
        update_pattern = update_pattern[:-1]
        info = filter_final_HORs[i]
        update_info = []
        for j in info:
            start = j[0]
            end = j[1]
            strand = j[2]
            sub_pattern = j[3]
            sub_pattern_repeat_number = j[4]
            update_sub_pattern = []
            for k in sub_pattern:
                update_sub_pattern.append(rename_monomer_ID[k])
            update_info.append([start, end, strand, update_sub_pattern, sub_pattern_repeat_number])
        update_filter_final_HORs[update_pattern] = update_info  # 添加了strand信息

    for i in top_layer:
        absolute_start = i[0]
        absolute_end = i[1]
        TD_item = i[2]
        TD_count = i[3]
        TD_range_index = i[4]
        relative_start = i[5]
        relative_end = i[6]
        updata_TD_item = []
        for j in TD_item:
            updata_TD_item.append(rename_monomer_ID[j])
        updata_TD_range_index = []
        for j in TD_range_index:
            start = j[0]
            end = j[1]
            sub_pattern = i[2]
            sub_pattern_repeat_number = i[3]
            updata_sub_pattern = []
            for k in sub_pattern:
                updata_sub_pattern.append(rename_monomer_ID[k])
            updata_TD_range_index.append([start, end, update_sub_pattern, sub_pattern_repeat_number])
        update_top_layer.append([absolute_start, absolute_end,
                                 updata_TD_item, TD_count, updata_TD_range_index,
                                 relative_start, relative_end])
    for i in all_layer:
        absolute_start = i[0]
        absolute_end = i[1]
        TD_item = i[2]
        TD_count = i[3]
        TD_range_index = i[4]
        relative_start = i[5]
        relative_end = i[6]
        updata_TD_item = []
        for j in TD_item:
            updata_TD_item.append(rename_monomer_ID[j])
        updata_TD_range_index = []
        for j in TD_range_index:
            start = j[0]
            end = j[1]
            sub_pattern = i[2]
            sub_pattern_repeat_number = i[3]
            updata_sub_pattern = []
            for k in sub_pattern:
                updata_sub_pattern.append(rename_monomer_ID[k])
            updata_TD_range_index.append([start, end, update_sub_pattern, sub_pattern_repeat_number])
        update_all_layer.append([absolute_start, absolute_end,
                                 updata_TD_item, TD_count, updata_TD_range_index,
                                 relative_start, relative_end])

    if len(pattern_score) == 0:
        statistics = [-10, -10, -10, -10, -10]
        return sk, update_cluster_block, update_monomer_sequence, \
               update_filter_final_HORs, statistics, update_top_layer, update_all_layer

    max_cov = -1
    max_cov_pattern = pattern_score[0][3].split('_')
    max_cov_pattern_monomer = set()
    for i in max_cov_pattern:
        max_cov_pattern_monomer.add(i)

    max_cov_rate = pattern_score[0][1]
    cov_rate = cov / len(monomer_sequence)
    max_pattern_score = pattern_score[0][0]

    if len(filter_final_HORs.keys()) == 0:
        CE_rate = -1
    else:
        CE_rate = len(max_cov_pattern_monomer) / len(max_cov_pattern)

    statistics = [cov_rate,
                  len(max_cov_pattern),
                  max_cov_rate,
                  CE_rate, max_pattern_score]
    # time_end = datetime.datetime.now()
    # print(str(similarity) + '\t' + str((time_end - time_start).seconds))

    buildHOR_time_end = datetime.datetime.now()
    log_file.write('minning Time: ' + str((buildHOR_time_end - buildHOR_time_start).seconds) + '\n')
    log_file.flush()

    return sk, update_cluster_block, update_monomer_sequence, \
           update_filter_final_HORs, \
           statistics, \
           update_top_layer, update_all_layer


# update, fix the problem cover start small than top
# 100,200,3_4,cover
# 100,100,3_4xxx,top
def outStatResult(K_output, outdir):
    out_statistics = outdir + '/out_statistics.xls'
    out_statistics = open(out_statistics, 'w')
    out_statistics.write('similarity' + '\t' +
                         'cov_rate' + '\t' +
                         'max_cov_pattern_len' + '\t' +
                         'max_cov_rate' + '\t' +
                         'CE_rate' + '\t' +
                         'max_pattern_score' + '\n')

    for k in K_output.keys():
        statistics = K_output[k][3]
        out_statistics.write(str(k) +
                             '\t' + str(statistics[0]) +
                             '\t' + str(statistics[1]) +
                             '\t' + str(statistics[2]) +
                             '\t' + str(statistics[3]) +
                             '\t' + str(statistics[4]) + '\n')
    out_statistics.close()


# out_final_hor 更新了strand
def outSingleResult(sk, cluster_block, monomer_sequence,
                    final_HOR, top_layer,
                    all_layer, outdir):
    # final_HOR 更新，增加了正负信息
    out_monomer_seq = outdir + '/out_monomer_seq_' + str(sk) + '.xls'
    out_monomer_seq = open(out_monomer_seq, 'w')
    out_monomer_seq.write(str(sk) + '\t')
    for i in monomer_sequence:
        out_monomer_seq.write(str(i) + ' ')
    out_monomer_seq.write('\n')
    out_monomer_seq.close()
    out_cluster = outdir + '/out_cluster_' + str(sk) + '.xls'
    out_cluster = open(out_cluster, 'w')
    for i in cluster_block.keys():
        out_cluster.write(str(i))
        for j in cluster_block[i]:
            out_cluster.write('\t' + str(j))
        out_cluster.write('\n')
    out_cluster.close()

    out_final_hor = outdir + '/out_final_hor' + str(sk) + '.xls'  # 更新：增加了正负信息
    out_final_hor = open(out_final_hor, 'w')
    for i in final_HOR.keys():
        out_final_hor.write(i + '\n')  # patter名字
        out_final_hor.write('P(start,end,strand,pattern,repeat_number):')
        for j in final_HOR[i]:
            out_final_hor.write('\t' + str(j[0]) + ',' + str(j[1]) + ',' + str(j[2]))
            pattern = ''
            for l in j[3]:
                pattern += str(l) + '_'
            pattern = pattern[:-1]
            out_final_hor.write(',' + pattern)
            out_final_hor.write(',' + str(j[4]))
        out_final_hor.write('\n')
    out_final_hor.close()

    # sort output
    top_layer_set = set()
    sorted_top_layer = sorted(top_layer, key=lambda x: x[0])
    out_top_layer = outdir + '/out_top_layer' + str(sk) + '.xls'
    out_top_layer = open(out_top_layer, 'w')

    layers = {}
    layers_info = {}
    for j in sorted_top_layer:
        start = j[0]
        end = j[1]
        pattern = ''
        repeat_number = j[3]
        for p in j[2]:
            pattern += str(p) + '_'
        pattern = pattern[:-1]
        top_layer_set.add(pattern + '_' + str(start) + '_' + str(end))
        out_top_layer.write(str(start) + '\t' + str(end) + '\t' + str(repeat_number) + '\t' + str(pattern) + '\n')
        layers[pattern + '_' + str(start) + '_' + str(end)] = []
        layers_info[pattern + '_' + str(start) + '_' + str(end)] = [start, end, repeat_number, pattern]
    out_top_layer.close()
    # fix same start region of top and cov
    sorted_all_layer = sorted(all_layer, key=lambda x: x[0])
    for j in sorted_all_layer:
        start = j[0]
        end = j[1]
        pattern = ''
        repeat_number = j[3]
        for p in j[2]:
            pattern += str(p) + '_'
        pattern = pattern[:-1]
        if pattern + '_' + str(start) + '_' + str(end) in layers.keys():
            pass
        else:
            for l in layers.keys():
                top_start = int(l.split('_')[-2])
                top_end = int(l.split('_')[-1])
                if start >= top_start and end <= top_end:
                    layers[l].append([start, end, repeat_number, pattern])

    out_all_layer = outdir + '/out_all_layer' + str(sk) + '.xls'
    out_all_layer = open(out_all_layer, 'w')
    for j in layers_info.keys():
        start = layers_info[j][0]
        end = layers_info[j][1]
        repeat_number = layers_info[j][2]
        pattern = layers_info[j][3]
        out_all_layer.write(
            str(start) + '\t' + str(end) + '\t' + str(repeat_number) + '\t' + str(pattern) + '\t' + 'top' + '\n')
        for k in layers[j]:
            sub_start = k[0]
            sub_end = k[1]
            sub_repeat_number = k[2]
            sub_pattern = k[3]
            out_all_layer.write(
                str(sub_start) + '\t' + str(sub_end) + '\t' + str(sub_repeat_number) + '\t' + str(
                    sub_pattern) + '\t' + 'cover' + '\n')
    out_all_layer.close()


def readPattern(pattern_file):
    patterns = {}
    key = ''
    with open(pattern_file, 'r') as pf:
        while True:
            line = pf.readline()[:-1]
            if not line:
                break
            key = line
            patterns[key] = []
            line = pf.readline()[:-1]
            itemsets = line.split('\t')[1:]
            # 增加strand
            for i in itemsets:
                r = i.split(',')
                start = int(r[0])
                end = int(r[1])
                strand = r[2]
                pattern = r[3]
                repeat_number = int(r[4])
                patterns[key].append([start, end, strand, pattern, repeat_number])
    return patterns


def readMonomerSequence(monomer_sequence_file, similarity):
    monomer_sequences = {}
    with open(monomer_sequence_file, 'r') as msf:
        while True:
            line = msf.readline()[:-2]
            if not line:
                break
            items = line.split('\t')
            monomer_sequence = items[1]
            monomer_sequences[items[0]] = monomer_sequence.split(' ')
    return monomer_sequences[similarity]


def readBlockSequence(block_sequence_file):
    block_sequence = []
    with open(block_sequence_file, 'r') as bsf:
        while True:
            line = bsf.readline()[:-1]
            if not line:
                break
            items = line.split('\t')
            for i in items:
                item = i.split('_')
                start = int(item[1])
                end = int(item[2])
                strand = item[3]
                block_sequence.append([start, end, strand])
    return block_sequence


def readCluster(cluster_file):
    monomer_table = {}
    with open(cluster_file, 'r') as cf:
        while True:
            line = cf.readline()[:-1]
            if not line:
                break
            items = line.split('\t')
            monomer_table[items[0]] = items[1:]
    return monomer_table


def buildMonomerFile(monomer_table, base_sequence, outdir):
    out_monomer = outdir + '/out_monomer.fa'

    out_monomer = open(out_monomer, 'w')
    for i in monomer_table.keys():
        count = 1
        database = monomer_table[i]
        for j in database:
            item = j.split('_')
            start = int(item[1])
            end = int(item[2])
            strand = item[3]
            out_monomer.write(
                '>' + str(i) + '.' + str(count) + '::' + str(start) + '-' + str(end) + ' ' + strand + '\n')
            out_monomer.write(base_sequence[start:end + 1])
            out_monomer.write('\n')
            count += 1
    out_monomer.close()


def buildHORFile(patterns, pattern_static, base_sequence, monomer_sequence, block_sequence, outdir):
    out_hor_raw_file = outdir + '/out_hor.raw.fa'
    out_hor_raw_file = open(out_hor_raw_file, 'w')
    out_hor_normal_file = outdir + '/out_hor.normal.fa'
    out_hor_normal_file = open(out_hor_normal_file, 'w')
    for i in patterns.keys():
        pattern_name = pattern_static[i][0]
        pattern = i.split('_')
        database = patterns[i]
        # ([start,end,strand,pattern,repeat_number])
        for j in database:
            start = j[0]
            end = j[1]
            strand = j[2]  # 更新增加strand
            monomer_sequence_item = monomer_sequence[start:end + 1]
            # patternname.index start end pattern repeatnumber rawpattern
            monomer_sequence_item_str = ''
            for k in monomer_sequence_item:
                monomer_sequence_item_str += k + '_'
            monomer_sequence_item_str = monomer_sequence_item_str[:-1]
            out_hor_raw_file.write('>' + pattern_name + '::' +
                                   str(block_sequence[start][0]) + '-' + str(block_sequence[end][1] + 1) +
                                   '::' + strand +
                                   ' nHOR-' + i + '::rHOR-' + monomer_sequence_item_str + '\n')
            out_hor_raw_file.write(base_sequence[block_sequence[start][0]:block_sequence[end][1] + 1] + '\n')

            out_hor_normal_file.write('>' + pattern_name + '::' +
                                      str(block_sequence[start][0]) + '-' + str(block_sequence[end][1] + 1) +
                                      '::' + strand +
                                      ' nHOR-' + i + '::rHOR-' + monomer_sequence_item_str + '\n')

            if len(pattern) == 1:
                # 考虑反链 '-' 链标准化变正
                normal_sequence = base_sequence[block_sequence[start][0]:block_sequence[end][1] + 1]
                if strand == '-':
                    normal_sequence = reverse(normal_sequence)
                out_hor_normal_file.write(normal_sequence + '\n')
            else:
                if strand == '-':
                    monomer_sequence_item = monomer_sequence_item[::-1]  # 反链序列翻转
                    double_sequence = monomer_sequence_item + monomer_sequence_item
                    double_index = list(range(len(monomer_sequence_item)))[::-1] + \
                                   list(range(len(monomer_sequence_item)))[::-1]  # 反链index翻转
                    count = 0
                    prefix = []
                    pattern_index = 0
                    for k in range(len(double_sequence)):
                        if pattern[pattern_index] == double_sequence[k]:
                            prefix.append([k, double_sequence[k], double_index[k], pattern_index])
                    normal_pattern = []
                    for k in prefix:
                        record = [k]
                        pattern_index = k[3] + 1
                        not_find = 0
                        for l in range(k[0] + 1, len(double_sequence)):
                            if double_sequence[l] == pattern[pattern_index]:
                                record.append([l, double_sequence[l], double_index[l], pattern_index])
                                pattern_index += 1
                                if pattern_index == len(pattern):
                                    break
                            else:
                                continue_flag = 0
                                for m in record:
                                    if double_sequence[l] == m[1]:
                                        continue_flag = 1
                                if continue_flag == 1:
                                    continue
                                else:
                                    not_find = 1
                                    break
                        if not_find == 1:
                            continue
                        if len(record) != len(pattern):
                            continue
                        normal_pattern = record
                    normal_sequence = ''
                    for k in normal_pattern:
                        block_start = block_sequence[start + k[2]][0]
                        block_end = block_sequence[start + k[2]][1] + 1
                        normal_sequence += reverse(base_sequence[block_start:block_end])  # 每个block变反
                    out_hor_normal_file.write(normal_sequence + '\n')
                else:
                    # +
                    double_sequence = monomer_sequence_item + monomer_sequence_item
                    double_index = list(range(len(monomer_sequence_item))) + list(range(len(monomer_sequence_item)))
                    count = 0
                    prefix = []
                    pattern_index = 0
                    for k in range(len(double_sequence)):
                        if pattern[pattern_index] == double_sequence[k]:
                            prefix.append([k, double_sequence[k], double_index[k], pattern_index])
                    normal_pattern = []
                    for k in prefix:
                        record = [k]
                        pattern_index = k[3] + 1
                        not_find = 0
                        for l in range(k[0] + 1, len(double_sequence)):
                            if double_sequence[l] == pattern[pattern_index]:
                                record.append([l, double_sequence[l], double_index[l], pattern_index])
                                pattern_index += 1
                                if pattern_index == len(pattern):
                                    break
                            else:
                                continue_flag = 0
                                for m in record:
                                    if double_sequence[l] == m[1]:
                                        continue_flag = 1
                                if continue_flag == 1:
                                    continue
                                else:
                                    not_find = 1
                                    break
                        if not_find == 1:
                            continue
                        if len(record) != len(pattern):
                            continue
                        normal_pattern = record
                    normal_sequence = ''
                    for k in normal_pattern:
                        block_start = block_sequence[start + k[2]][0]
                        block_end = block_sequence[start + k[2]][1] + 1
                        normal_sequence += base_sequence[block_start:block_end]
                    out_hor_normal_file.write(normal_sequence + '\n')
    out_hor_raw_file.close()
    out_hor_normal_file.close()


def Plot(monomer_sequence, patterns, pattern_static, block_seuqence, outdir, show_number=5, show_min_repeat_number=10):
    fig, ax = plt.subplots(figsize=(10, 10))
    monomer_len = len(monomer_sequence)

    color = '#D14524'
    custom_lines = []
    legend_text = []

    filter_patterns = {}
    pattern_count = 0
    for i in patterns.keys():
        if pattern_count >= show_number:
            break
        pattern_name = pattern_static[i][0]
        pattern_repeat_number = pattern_static[i][1]
        if pattern_repeat_number < show_min_repeat_number:
            continue
        filter_patterns[i] = patterns[i]
        pattern_count += 1

    re_patterns = list(filter_patterns.keys())[::-1]
    pattern_count = 0
    for i in re_patterns:
        # print(pattern_static[i])
        pattern_name = pattern_static[i][0]
        xy = np.array([0, pattern_count * monomer_len / 25])
        rect = mpathes.Rectangle(xy, monomer_len, monomer_len / 50, color='#D0CECE')
        ax.add_patch(rect)
        custom_lines.append(Line2D([0], [0], color=color, lw=2))
        legend_text.append(i)
        for j in patterns[i]:
            start = j[0]
            end = j[1]
            xy2 = np.asarray([start, pattern_count * monomer_len / 25])
            rect = mpathes.Rectangle(xy2, end + 1 - start, monomer_len / 50, color=color, lw=0)
            ax.add_patch(rect)
        plt.text(monomer_len + monomer_len / 50, pattern_count * monomer_len / 25, pattern_name, fontsize=10)
        pattern_count += 1

    xy3 = np.asarray([0, -monomer_len / 50])
    rect = mpathes.Rectangle(xy3, monomer_len, monomer_len / 1000, color='black')
    ax.add_patch(rect)
    point_bar = int(monomer_len / 10)
    for i in range(10):
        xy3 = np.asarray([0 + i * point_bar, -monomer_len / 50])
        rect = mpathes.Rectangle(xy3, monomer_len / 1000, -monomer_len / 100, color='black')
        ax.add_patch(rect)
        plt.text(0 + i * point_bar, -monomer_len / 50 - monomer_len / 50, str(block_seuqence[0 + i * point_bar][0]),
                 fontsize=5)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.legend(custom_lines,legend_text)
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.savefig(outdir + '/plot_pattern.pdf')
    plt.close()

    x = np.arange(min(len(filter_patterns.keys()), show_number))
    y = []
    y1 = []

    bar_width = 0.35

    tick_label = []
    for i in filter_patterns.keys():
        database = patterns[i]
        pattern = i.split('_')
        canonical = 0
        nested = 0
        for j in database:
            item_len = int(j[1]) + 1 - int(j[0])
            if item_len == len(pattern) * j[4]:
                canonical += j[4]
            else:
                nested += j[4]
        y.append(canonical)
        y1.append(nested)
        tick_label.append(pattern_static[i][0])

    pattern_static_file = outdir + '/pattern_static.xls'
    pattern_static_file = open(pattern_static_file, 'w')
    pattern_static_file.write('HORs\tCanonical\tNested\n')

    for i in range(len(tick_label)):
        pattern_static_file.write(tick_label[i] + '\t' + str(y[i]) + '\t' + str(y1[i]) + '\n')
    pattern_static_file.close()

    plt.figure(figsize=(10, 10))
    plt.bar(x, y, bar_width, align="center", color="c", label="canonical", alpha=0.5)
    plt.bar(x + bar_width, y1, bar_width, color="b", align="center", label="nested", alpha=0.5)

    plt.xlabel("HORs")
    plt.ylabel("Repeat Number")

    plt.xticks(x + bar_width / 2, tick_label)

    plt.legend()

    plt.savefig(outdir + '/pattern_static.pdf')
    plt.close()


def readTopLayer(top_layer_file):
    top_layer = []
    with open(top_layer_file, 'r') as tf:
        while True:
            line = tf.readline()[:-1]
            if not line:
                break
            items = line.split('\t')
            top_layer.append(items)
    return top_layer


def readAllLayer(all_layer_file):
    all_layer = []
    with open(all_layer_file, 'r') as tf:
        while True:
            line = tf.readline()[:-1]
            if not line:
                break
            items = line.split('\t')
            all_layer.append(items)
    return all_layer


def getBestResult(base_sequence, outdir, show_hor_number, show_hor_min_repeat_number):
    outdir_best = outdir + '/out'
    if not os.path.exists(outdir_best):
        os.mkdir(outdir_best)
    statistic_file = outdir + '/out_statistics.xls'
    statistic = pd.read_csv(statistic_file, sep='\t')
    statistic = statistic.sort_values(by=['cov_rate', 'max_cov_rate', 'similarity'], ascending=False)[
        ['cov_rate', 'max_cov_rate', 'similarity']]
    statistic.to_csv(outdir_best + '/sort_statistics.xls', sep='\t', index=None)
    statistic = np.asarray(statistic)
    similarity = str(int(statistic[0][-1]))
    pattern_file = outdir + '/out_final_hor' + similarity + '.xls'  # 存在更新
    cluster_file = outdir + '/out_cluster_' + similarity + '.xls'
    monomer_sequence_file = outdir + '/out_monomer_seq_' + similarity + '.xls'
    block_sequence_file = outdir + '/out_block.sequences'
    pattern_repeat_file = outdir_best + '/hor.repeatnumber.xls'

    patterns = readPattern(pattern_file)  # 更新增加strand [start,end,strand,pattern,repeat_number]
    monomer_sequence = readMonomerSequence(monomer_sequence_file, similarity)
    block_sequence = readBlockSequence(block_sequence_file)

    pattern_static = {}
    pattern_index = 1

    pattern_repeat_file = open(pattern_repeat_file, 'w')
    pattern_repeat_file.write('HORs\tRepeatNumber\n')

    for i in patterns.keys():
        pattern = i.split('_')
        database = patterns[i]  # 增加了strand
        repeat_number = 0
        for j in database:
            repeat_number += j[4]
        pattern_name = 'R' + str(pattern_index) + 'L' + str(len(pattern))
        pattern_repeat_file.write(pattern_name + '\t' + str(repeat_number) + '\n')
        pattern_static[i] = [pattern_name, repeat_number]
        pattern_index += 1
    pattern_repeat_file.close()

    all_layer = {}

    for seq in pattern_static.keys():
        all_layer[seq] = []

    all_layer_file = outdir + '/out_all_layer' + similarity + '.xls'
    with open(all_layer_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            start = line[0]
            end = line[1]
            count = line[2]
            td_monomer_pattern = line[3]
            reverse_td_monomer_pattern = "_".join(td_monomer_pattern.split('_')[::-1])
            for seq in patterns.keys():
                if len(seq) != len(td_monomer_pattern):
                    continue
                db_seq = seq + '_' + seq
                if td_monomer_pattern in db_seq:
                    if seq in all_layer.keys():
                        all_layer[seq].append([int(start), int(end), '+', td_monomer_pattern, int(count)])
                    else:
                        all_layer[seq] = [[int(start), int(end), '+', td_monomer_pattern, int(count)]]
                elif reverse_td_monomer_pattern in db_seq:
                    if seq in all_layer.keys():
                        all_layer[seq].append([int(start), int(end), '+', td_monomer_pattern, int(count)])
                    else:
                        all_layer[seq] = [[int(start), int(end), '+', td_monomer_pattern, int(count)]]

    Plot(monomer_sequence, all_layer, pattern_static, block_sequence, outdir_best,
         show_number=show_hor_number, show_min_repeat_number=show_hor_min_repeat_number)

    monomer_table = readCluster(cluster_file)
    buildMonomerFile(monomer_table, base_sequence, outdir_best)
    buildHORFile(patterns, pattern_static, base_sequence, monomer_sequence, block_sequence, outdir_best)

    top_layer_file = outdir + '/out_top_layer' + similarity + '.xls'
    all_layer_file = outdir + '/out_all_layer' + similarity + '.xls'
    top_layer = readTopLayer(top_layer_file)
    all_layer = readAllLayer(all_layer_file)
    # add name
    # print(pattern_static)

    new_top_layer = []
    for i in top_layer:
        start = int(i[0])
        end = int(i[1])
        repeat_number = i[2]
        pattern = i[3].split('_')
        start_block = block_sequence[start]
        strand = start_block[-1]
        in_flag = 0
        if strand == '-':
            pattern = pattern[::-1]
        for j in range(len(pattern)):
            prefix_pattern = pattern[j:]
            suffix_pattern = pattern[:j]
            loop_pattern = prefix_pattern + suffix_pattern
            s_loop_pattern = ''
            for k in loop_pattern:
                s_loop_pattern += str(k) + '_'
            s_loop_pattern = s_loop_pattern[:-1]
            if s_loop_pattern in pattern_static.keys():
                new_top_layer.append([start, end, repeat_number, i[3], pattern_static[s_loop_pattern][0]])
                break
    out_top_layer_file = outdir_best + '/out_top_layer.xls'
    out_top_layer_file = open(out_top_layer_file, 'w')
    for i in new_top_layer:
        out_top_layer_file.write(
            i[4] + '\t' + str(block_sequence[i[0]][0]) + '\t' + str(block_sequence[i[1]][1]) + '\t' + i[2] + '\t' + i[
                3] + '\n')
    out_top_layer_file.close()

    new_all_layer = []
    for i in all_layer:
        start = int(i[0])
        end = int(i[1])
        repeat_number = i[2]
        pattern = i[3].split('_')
        start_block = block_sequence[start]
        strand = start_block[-1]
        type = i[4]
        in_flag = 0
        if strand == '-':
            pattern = pattern[::-1]
        for j in range(len(pattern)):
            prefix_pattern = pattern[j:]
            suffix_pattern = pattern[:j]
            loop_pattern = prefix_pattern + suffix_pattern
            s_loop_pattern = ''
            for k in loop_pattern:
                s_loop_pattern += str(k) + '_'
            s_loop_pattern = s_loop_pattern[:-1]
            if s_loop_pattern in pattern_static.keys():
                new_all_layer.append([start, end, repeat_number, i[3], pattern_static[s_loop_pattern][0], type])
                break
    out_all_layer_file = outdir_best + '/out_all_layer.xls'
    out_all_layer_file = open(out_all_layer_file, 'w')
    for i in new_all_layer:
        out_all_layer_file.write(
            i[4] + '\t' + str(block_sequence[i[0]][0]) + '\t' + str(block_sequence[i[1]][1]) + '\t' + i[2] + '\t' + i[
                3] + '\t' + i[5] + '\n')
    out_all_layer_file.close()


def main():
    all_time_start = datetime.datetime.now()

    parser = argparse.ArgumentParser(description="Mining HOR")
    parser.add_argument("-d", "--decomposition_path")
    parser.add_argument("-b", "--base_sequence_path")
    parser.add_argument("-o", "--outdir")
    parser.add_argument("-s", "--min_similarity", type=float, default=0.94)
    parser.add_argument("-st", "--step", type=float, default=0.005)
    parser.add_argument("-m", "--max_hor_len", type=int, default=40)
    parser.add_argument("-t", "--thread", type=int, default=1)
    parser.add_argument("-sp", "--show_hor_number", type=int, default=5)
    parser.add_argument("-sn", "--show_hor_min_repeat_number", type=int, default=10)

    args = parser.parse_args()

    decomposition_path = args.decomposition_path
    base_sequence_path = args.base_sequence_path
    outdir = args.outdir
    min_similarity = args.min_similarity
    step = args.step
    max_hor_len = args.max_hor_len
    thread = args.thread

    show_hor_number = args.show_hor_number
    show_hor_min_repeat_number = args.show_hor_min_repeat_number

    log_file = outdir + '/log.txt'
    log_file = open(log_file, 'w')

    print('start')

    print('build block sequence and read base sequence')
    block_sequence = buildMonomerBlockSequence(decomposition_path)
    base_sequence = ''
    with open(base_sequence_path, 'r') as f:
        f.readline()
        base_sequence = f.readline()[:-1]

    print('calculate ed distance')
    print('ed distance thread: ' + str(thread))
    ed_time_start = datetime.datetime.now()
    edit_distance_matrix, block_name_index = calculateED(block_sequence, base_sequence, thread)
    ed_time_end = datetime.datetime.now()
    log_file.write('ed Time: ' + str((ed_time_end - ed_time_start).seconds) + '\n')
    log_file.flush()

    # edit_distance_matrix = pd.read_csv(outdir + '/out_edit_distance_matrix.xls',sep='\t',index_col=0)
    # block_name_index = edit_distance_matrix.index.tolist()
    # edit_distance_matrix = np.asarray(edit_distance_matrix)

    block_name_table = {}
    for i in range(len(block_name_index)):
        block_name_table[block_name_index[i]] = i

    out_edit_distance_matrix = pd.DataFrame(edit_distance_matrix,
                                            columns=block_name_index,
                                            index=block_name_index)
    # out_edit_distance_matrix.to_csv(outdir + '/out_edit_distance_matrix.xls', sep='\t')
    print(len(edit_distance_matrix))
    # print(edit_distance_matrix)

    out_block_sequence_path = outdir + '/out_block.sequences'
    out_block_sequence_path = open(out_block_sequence_path, 'w')
    str_line = ''
    for i in block_sequence:
        str_line += i + '\t'
    out_block_sequence_path.write(str_line[:-1] + '\n')
    out_block_sequence_path.close()

    print('pre merge matrix')
    merge_edit_distance_matrix, merge_block_name_index, final_pre_merge = pre_Clustering(edit_distance_matrix,
                                                                                         block_name_index)
    print(len(merge_edit_distance_matrix))

    out_merge_edit_distance_matrix = pd.DataFrame(merge_edit_distance_matrix, columns=merge_block_name_index,
                                                  index=merge_block_name_index)
    # out_merge_edit_distance_matrix.to_csv(outdir + '/out_merge_edit_distance_matrix.xls', sep='\t')

    out_final_pre_merge_path = outdir + '/out_pre_merge.xls'
    out_final_pre_merge_path = open(out_final_pre_merge_path, 'w')
    for i in final_pre_merge.keys():
        out_final_pre_merge_path.write(str(i))
        for j in final_pre_merge[i]:
            out_final_pre_merge_path.write('\t' + str(j))
        out_final_pre_merge_path.write('\n')
    out_final_pre_merge_path.close()

    print('generation cover distribution for cluster')
    K_output = {}

    K = int((1 - min_similarity) / step)

    k_list = [i for i in range(0, K + 1)]
    # print(k_list)
    print('HOR thread: ' + str(1))
    result = []
    for k in k_list:
        log_file.write('k: ' + str(k) + '\n')
        log_file.flush()
        sk, update_cluster_block, update_monomer_sequence, \
        update_filter_final_HORs, \
        statistics, \
        update_top_layer, update_all_layer = clusterAndFindingHOR(k, min_similarity, step,
                                                                  merge_edit_distance_matrix,
                                                                  merge_block_name_index, block_sequence,
                                                                  final_pre_merge, block_name_index, max_hor_len,
                                                                  log_file)
        result.append([sk, update_cluster_block,
                       update_monomer_sequence,
                       update_filter_final_HORs,
                       statistics, update_top_layer, update_all_layer])
        outSingleResult(sk, update_cluster_block, update_monomer_sequence,
                        update_filter_final_HORs, update_top_layer,
                        update_all_layer, outdir)

    print('get result')
    # 0 sk,
    # 1 update_cluster_block,
    # 2 update_monomer_sequence,
    # 3 update_filter_final_HORs, *
    # 4 statistics
    # 5 update_top_layer
    # 6 update_all_layer
    for i in result:
        K_output[i[0]] = [i[1], i[2], i[3], i[4], i[5], i[6]]

    outStatResult(K_output, outdir)

    getBestResult(base_sequence, outdir, show_hor_number, show_hor_min_repeat_number)

    all_time_end = datetime.datetime.now()
    print('Time: ' + str((all_time_end - all_time_start).seconds))
    log_file.close()


if __name__ == '__main__':
    main()

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.lines import Line2D

def reverse(sequence):
    base_map = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
    new_sequence = ''
    for i in sequence[::-1]:
        new_sequence += base_map[i]
    return new_sequence

def readPattern(pattern_file):
    patterns = {}
    key = ''
    with open(pattern_file,'r') as pf:
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
                patterns[key].append([start,end,strand,pattern,repeat_number])
    return patterns

def readMonomerSequence(monomer_sequence_file, similarity):
    monomer_sequences = {}
    with open(monomer_sequence_file,'r') as msf:
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
    with open(block_sequence_file,'r') as bsf:
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
                block_sequence.append([start,end,strand])
    return block_sequence

def readCluster(cluster_file):
    monomer_table = {}
    with open(cluster_file,'r') as cf:
        while True:
            line = cf.readline()[:-1]
            if not line:
                break
            items = line.split('\t')
            monomer_table[items[0]] = items[1:]
    return monomer_table

def buildMonomerFile(monomer_table,base_sequence,outdir):
    out_monomer = outdir + '/out_monomer.fa'

    out_monomer = open(out_monomer,'w')
    for i in monomer_table.keys():
        count = 1
        database = monomer_table[i]
        for j in database:
            item = j.split('_')
            start = int(item[1])
            end = int(item[2])
            strand = item[3]
            out_monomer.write('>' + str(i) + '.' + str(count) + '::' +str(start) +'-' + str(end) +' ' + strand + '\n')
            out_monomer.write(base_sequence[start:end+1])
            out_monomer.write('\n')
            count += 1
    out_monomer.close()

def buildHORFile(patterns, pattern_static,base_sequence,monomer_sequence,block_sequence,outdir):
    out_hor_raw_file = outdir + '/out_hor.raw.fa'
    out_hor_raw_file = open(out_hor_raw_file,'w')
    out_hor_normal_file = outdir + '/out_hor.normal.fa'
    out_hor_normal_file = open(out_hor_normal_file,'w')
    for i in patterns.keys():
        pattern_name = pattern_static[i][0]
        pattern = i.split('_')
        database = patterns[i]
        # ([start,end,strand,pattern,repeat_number])
        for j in database:
            start = j[0]
            end = j[1]
            strand = j[2] # 更新增加strand
            monomer_sequence_item = monomer_sequence[start:end+1]
            # patternname.index start end pattern repeatnumber rawpattern
            monomer_sequence_item_str = ''
            for k in monomer_sequence_item:
                monomer_sequence_item_str += k + '_'
            monomer_sequence_item_str = monomer_sequence_item_str[:-1]
            out_hor_raw_file.write('>' + pattern_name  + '::' +
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
                    monomer_sequence_item = monomer_sequence_item[::-1] # 反链序列翻转
                    double_sequence = monomer_sequence_item + monomer_sequence_item
                    double_index = list(range(len(monomer_sequence_item)))[::-1] + \
                                   list(range(len(monomer_sequence_item)))[::-1] # 反链index翻转
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
                        normal_sequence += reverse(base_sequence[block_start:block_end]) # 每个block变反
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
                        block_end = block_sequence[start + k[2]][1]+1
                        normal_sequence += base_sequence[block_start:block_end]
                    out_hor_normal_file.write(normal_sequence + '\n')
    out_hor_raw_file.close()
    out_hor_normal_file.close()

def Plot(monomer_sequence, patterns,pattern_static, block_seuqence, outdir, show_number = 5, show_min_repeat_number = 10):
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
            rect = mpathes.Rectangle(xy2, end + 1 - start, monomer_len / 50, color=color,lw=0)
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
        plt.text(0 + i * point_bar, -monomer_len / 50 - monomer_len / 50, str(block_seuqence[0 + i * point_bar][0]), fontsize=5)

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

    x = np.arange(min(len(filter_patterns.keys()),show_number))
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
    pattern_static_file = open(pattern_static_file,'w')
    pattern_static_file.write('HORs\tCanonical\tNested\n')

    for i in range(len(tick_label)):
        pattern_static_file.write(tick_label[i]+'\t'+str(y[i]) +'\t' +str(y1[i]) + '\n')
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
    with open(top_layer_file,'r') as tf:
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


def getResult(similarity,base_sequence,result_dir,show_hor_number,show_hor_min_repeat_number):
    outdir_best = result_dir + '/out_' + similarity
    if not os.path.exists(outdir_best):
        os.mkdir(outdir_best)

    pattern_file = result_dir + '/out_final_hor'+similarity+'.xls'
    cluster_file = result_dir + '/out_cluster_'+similarity+'.xls'
    monomer_sequence_file = result_dir + '/out_monomer_seq_'+similarity+'.xls'
    block_sequence_file = result_dir + '/out_block.sequences'
    pattern_repeat_file = outdir_best + '/hor.repeatnumber.xls'

    patterns = readPattern(pattern_file)
    monomer_sequence = readMonomerSequence(monomer_sequence_file, similarity)
    block_sequence = readBlockSequence(block_sequence_file)


    pattern_static = {}
    pattern_index = 1

    pattern_repeat_file = open(pattern_repeat_file,'w')
    pattern_repeat_file.write('HORs\tRepeatNumber\n')

    for i in patterns.keys():
        pattern = i.split('_')
        database = patterns[i]
        repeat_number = 0
        for j in database:
            repeat_number += j[4]
        pattern_name = 'R'+str(pattern_index) + 'L' + str(len(pattern))
        pattern_repeat_file.write(pattern_name+'\t'+str(repeat_number) + '\n')
        pattern_static[i] = [pattern_name,repeat_number]
        pattern_index += 1
    pattern_repeat_file.close()

    all_layer = {}

    for seq in pattern_static.keys():
        all_layer[seq] = []

    all_layer_file = result_dir + '/out_all_layer' + similarity + '.xls'
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
         show_number=show_hor_number,show_min_repeat_number=show_hor_min_repeat_number)

    monomer_table = readCluster(cluster_file)
    buildMonomerFile(monomer_table, base_sequence, outdir_best)
    buildHORFile(patterns, pattern_static,base_sequence,monomer_sequence,block_sequence,outdir_best)

    top_layer_file = result_dir + '/out_top_layer' + similarity + '.xls'
    all_layer_file = result_dir + '/out_all_layer' + similarity + '.xls'
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
    parser = argparse.ArgumentParser(description="Get given similarity HORs")
    parser.add_argument("-r", "--result_dir",help="HiCAT result path, required",required=True)
    parser.add_argument("-s", "--similarity",help="Given similarity, required",required=True)
    parser.add_argument("-sp", "--show_hor_number", help="Default visualized the top five HORs", type=int, default=5,required=False)
    parser.add_argument("-sn", "--show_hor_min_repeat_number", help="Default visualized the HORs with repeat numbers greater than 10", type=int, default=10,required=False)
    args = parser.parse_args()

    result_dir = args.result_dir
    similarity = args.similarity

    show_hor_number = args.show_hor_number
    show_hor_min_repeat_number = args.show_hor_min_repeat_number


    base_sequence_path = result_dir + '/' + 'input_fasta.1.fa'
    base_sequence = ''
    with open(base_sequence_path, 'r') as f:
        f.readline()
        base_sequence = f.readline()[:-1]

    getResult(similarity,base_sequence,result_dir,show_hor_number,show_hor_min_repeat_number)

if __name__ == '__main__':
    main()
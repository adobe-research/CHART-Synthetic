
import os
import sys
import json
import math
import itertools
import editdistance
import numpy as np
import scipy.optimize
import scipy.spatial.distance


def euclid(p1, p2):
    x1 = float(p1['x'])
    y1 = float(p1['y'])
    x2 = float(p2['x'])
    y2 = float(p2['y'])
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# def box_to_discrete(ds):
#     out = []
#     for it_name in ['first_quartile', 'max', 'min', 'median', 'third_quartile']: 
#         out.append( {'name': it_name, 'x': ds[it_name]['x'], 'y': ds[it_name]['y']} )
#     return out

def box_arr_to_np(ds):
    n = np.zeros( (len(ds), 8))
    for i,p in enumerate(ds):
        cnt_q = 0
        for quart in p:        
            n[i,cnt_q] = float(p[quart]['y'])
            n[i,cnt_q+1] = float(p[quart]['x'])
            cnt_q = cnt_q+1
    return n

def compare_box(pred_ds, gt_ds, min_dim):
    pred_ds = box_arr_to_np(pred_ds)
    gt_ds = box_arr_to_np(gt_ds)
    cost_mat = np.minimum(1, scipy.spatial.distance.cdist(pred_ds, gt_ds, metric='cityblock') /(min_dim*0.1))
    return get_score(cost_mat)

def arr_to_np(ds):
    n = np.zeros((len(ds), 2))
    for i,p in enumerate(ds):
        n[i,0] = float(p['x'])
        n[i,1] = float(p['y'])
    return n

def bar_arr_to_np(ds):
    n = np.zeros( (len(ds), 4))
    for i,p in enumerate(ds):
        n[i,0] = float(p['y0'])
        n[i,1] = float(p['x0'])
        n[i,2] = float(p['height']) + float(p['y0'])
        n[i,3] = float(p['width']) + float(p['x0'])
    return n

def compare_bar(pred_ds, gt_ds, min_dim):
    pred_ds = bar_arr_to_np(pred_ds)
    gt_ds = bar_arr_to_np(gt_ds)

    cost_mat = np.minimum(1, scipy.spatial.distance.cdist(pred_ds, gt_ds, metric='cityblock') /(min_dim(min_dim*0.1)))
    return get_score(cost_mat)

def compare_scatter(pred_ds, gt_ds, gamma):
    pred_ds = arr_to_np(pred_ds)
    gt_ds = arr_to_np(gt_ds)

    V = np.cov(gt_ds.T)
    VI = np.linalg.inv(V).T

    cost_mat = np.minimum(1, scipy.spatial.distance.cdist(pred_ds, gt_ds, metric='mahalanobis', VI=VI) / gamma)
    #print(cost_mat)
    return get_score(cost_mat)

def get_score(cost_mat):
    cost_mat = pad_mat(cost_mat)
    k = cost_mat.shape[0]

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind].sum()
    score = 1 - (cost / k)
    return score

def get_cont_recall(p_xs, p_ys, g_xs, g_ys, epsilon):
    total_score = 0
    total_interval = 0
    for i in range(g_xs.shape[0]):
        x = g_xs[i]
        if i == 0:
            interval = (g_xs[i+1] - x) / 2
        elif i == (g_xs.shape[0] - 1):
            interval = (x - g_xs[i-1]) / 2
        else:
            interval = (g_xs[i+1] - g_xs[i-1]) / 2

        y = g_ys[i]
        y_interp = np.interp(x, p_xs, p_ys)
        error = min(1, abs( (y - y_interp) / (abs(y) + epsilon)))
        total_score += (1 - error) * interval
        total_interval += interval
    assert np.isclose(total_interval, g_xs[-1] - g_xs[0])
    return total_score / total_interval

def compare_continuous(pred_ds, gt_ds):
    pred_ds = sorted(pred_ds, key=lambda p: float(p['x']))
    gt_ds = sorted(gt_ds, key=lambda p: float(p['x']))
    p_xs = np.array([float(ds['x']) for ds in pred_ds])
    p_ys = np.array([float(ds['y']) for ds in pred_ds])
    g_xs = np.array([float(ds['x']) for ds in gt_ds])
    g_ys = np.array([float(ds['y']) for ds in gt_ds])

    epsilon = (g_ys.max() - g_ys.min()) / 100.
    recall = get_cont_recall(p_xs, p_ys, g_xs, g_ys, epsilon)
    precision = get_cont_recall(g_xs, g_ys, p_xs, p_ys, epsilon)

    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.

def norm_edit_dist(s1, s2):
    return editdistance.eval(s1, s2) / float(max(len(s1), len(s2), 1))

def create_dist_mat(seq1, seq2, compare):
    l1 = len(seq1)
    l2 = len(seq2)
    mat = np.full( (l1, l2), -1.)
    for i in range(l1):
        for j in range(l2):
            mat[i,j] = compare(seq1[i], seq2[j])
    return mat

def pad_mat(mat):
    h,w = mat.shape
    if h == w:
        return mat
    elif h > w:
        new_mat = np.ones( (h, h) )
        new_mat[:,:w] = mat
        return new_mat
    else:
        new_mat = np.ones( (w, w) )
        new_mat[:h,:] = mat
        return new_mat


def metric_6a(pred_data_series, gt_data_series, gt_type, alpha=1, beta=2, gamma=1, img_dim = [1280.0, 960.0], debug=False):
    if 'box' in gt_type.lower():
        compare = lambda ds1, ds2: compare_box(ds1, ds2, min(img_dim))
        pred_no_names = list(map(lambda ds: ds['boxplots'], pred_data_series))
        gt_no_names = list(map(lambda ds: ds['boxplots'], gt_data_series))
    if 'bar' in gt_type.lower():
        compare = lambda ds1, ds2: compare_bar(ds1, ds2, min(img_dim))
        pred_no_names = list(map(lambda ds: ds['bars'], pred_data_series))
        gt_no_names = list(map(lambda ds: ds['bars'], gt_data_series))
    elif 'scatter' in gt_type.lower():
        compare = lambda ds1, ds2: compare_scatter(ds1, ds2, gamma)
        pred_no_names = list(map(lambda ds: ds['scatter points'], pred_data_series))
        gt_no_names = list(map(lambda ds: ds['scatter points'], gt_data_series))
    elif 'line' in gt_type.lower():
        compare = compare_continuous
        pred_no_names = list(map(lambda ds: ds['lines'], pred_data_series))
        gt_no_names = list(map(lambda ds: ds['lines'], gt_data_series))
    else:
        raise Exception("Odd Case")
  
    ds_match_scores = create_dist_mat(pred_no_names, gt_no_names, compare)
    if debug:
        print("Data Series Match Scores:")
        print(ds_match_scores)

    mat1 = 1 - (ds_match_scores / beta)
    cost_mat = mat1
    if debug:
        print("\nCost Matrix:")
        print(cost_mat)
    return get_score(cost_mat)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: python metric6a.py pred_file|pred_dir gt_file|gt_dir [alpha] [beta] [gamma] [img_dim] [debug]")
        exit()
    pred_infile = sys.argv[1]
    gt_infile = sys.argv[2]

    try:
        alpha = float(sys.argv[3])
    except:
        alpha = 1
    try:
        beta = float(sys.argv[4])
    except:
        beta = 2
    try:
        gamma = float(sys.argv[5])
    except:
        gamma = 1    
    try:
        img_dim = sys.argv[6]
    except:
        img_dim = [1280, 960.0]
    try:
        debug = sys.argv[7]
    except:
        debug = False

    if os.path.isfile(pred_infile) and os.path.isfile(gt_infile):
        pred_json = json.load(open(pred_infile))
        gt_json = json.load(open(gt_infile))

        pred_outputs = pred_json['task6']['output']['visual elements']
        gt_outputs = gt_json['task6']['output']['visual elements']
        gt_type = gt_json['task1']['output']['chart_type']

        score = metric_6a(pred_outputs, gt_outputs, gt_type, alpha, beta, gamma, img_dim, debug)
        print(score)
    elif os.path.isdir(pred_infile) and os.path.isdir(gt_infile):
        scores = []
        for x in os.listdir(pred_infile):
            pred_file = os.path.join(pred_infile, x)
            gt_file = os.path.join(gt_infile, x)

            pred_json = json.load(open(pred_file))
            gt_json = json.load(open(gt_file))

            pred_outputs = pred_json['task6']['output']['visual elements']
            gt_outputs = gt_json['task6']['output']['visual elements']
            gt_type = gt_json['task1']['output']['chart_type']

            score = metric_6a(pred_outputs, gt_outputs, gt_type, alpha, beta, gamma, img_dim, debug)
            scores.append(score)
            print("%s: %f" % (x, score))
        avg_score = sum(scores) / len(scores)
        print("Average Score: %f" % avg_score)
    else:
        print("Error: pred_file and gt_file must both be files or both be directories")
        exit()



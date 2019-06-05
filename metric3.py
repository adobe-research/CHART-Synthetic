import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

ATEAM_LABELS = ['legend_label', 'chart_title', 'tick_label', 'axis_title', 'other', 'legend_title']
SYNTH_LABELS = ['legend_label', 'chart_title', 'tick_label', 'axis_title']


def get_confusion_matrix(confusion, unique_labels):
    label_idx_map = {label : i for i, label in enumerate(unique_labels)}
    idx_label_map = {i : label for label, i in label_idx_map.items()}
    cmat = np.zeros((len(label_idx_map), len(label_idx_map)))
    for ID, pair in confusion.items():
        truth, pred = pair
        if pred is None or pred not in label_idx_map:
            continue
        if truth not in label_idx_map:
            continue
        t = label_idx_map[truth]
        p = label_idx_map[pred]
        cmat[t, p] += 1
    norm = cmat.sum(axis=1).reshape(-1, 1)
    cmat /= norm
    return cmat, idx_label_map


def plot_confusion_matrix(cm, classes, output_img_path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    fig.savefig(output_img_path, bbox_inches='tight')
    plt.show()

def eval_task3(gt_folder, result_folder, output_img_path):
    gt_label_map = {}
    result_label_map = {}
    metrics = {}
    confusion = {}
    ignore = set()
    result_files = os.listdir(result_folder)
    gt_files = os.listdir(gt_folder)
    for gt_file in gt_files:
        gt_id = ''.join(gt_file.split('.')[:-1])
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)
        text_roles = gt['task3']['output']['text_roles']
        for text_role in text_roles:
            text_id = text_role['id']
            role = text_role['role'].lower().strip()
            # SOME LABELS IN PMC NOT PRESENT IN SYNTHETIC, TO BE CONSIDERED AS DONT CARE FOR EVAL
            if role not in SYNTH_LABELS:
                ignore.add('{}__sep__{}'.format(gt_id, text_id))
                continue
            gt_label_map[role] = gt_label_map[role] + ['{}__sep__{}'.format(gt_id, text_id)] \
                if role in gt_label_map else ['{}__sep__{}'.format(gt_id, text_id)]
            confusion['{}__sep__{}'.format(gt_id, text_id)] = [role, None]
    print(len(ignore))
    unique_roles = set()
    for result_file in result_files:
        result_id = ''.join(result_file.split('.')[:-1])
        with open(os.path.join(result_folder, result_file), 'r') as f:
            result = json.load(f)
        try:
            if 'text_roles' in result['task3']['output']:
                text_roles = result['task3']['output']['text_roles']
            # this is due to wrong json format in a submission
            else:
                text_roles = result['task3']['output']['text_blocks']
            for text_role in text_roles:
                text_id = text_role['id']
                role = text_role['role'].lower().strip()
                # SOME LABELS IN PMC NOT PRESENT IN SYNTHETIC, TO BE CONSIDERED AS DONT CARE FOR EVAL
                unique_roles.add(role)
                if '{}__sep__{}'.format(result_id, text_id) in ignore:
                    continue
                result_label_map[role] = result_label_map[role] + ['{}__sep__{}'.format(result_id, text_id)]\
                    if role in result_label_map else ['{}__sep__{}'.format(result_id, text_id)]
                confusion['{}__sep__{}'.format(result_id, text_id)][1] = role
        except Exception as e:
            print(e)
            print('invalid result json format in {} please check against provided samples'.format(result_file))
            continue
    total_recall = 0.
    total_precision = 0.
    total_fmeasure = 0.

    # print(unique_roles)

    for label, gt_instances in gt_label_map.items():
        res_instances = set(result_label_map[label])
        gt_instances = set(gt_instances)
        intersection = gt_instances.intersection(res_instances)
        print(label, len(gt_instances), len(res_instances), len(intersection))
        recall = len(intersection) / float(len(gt_instances))
        precision = len(intersection) / float(len(res_instances))
        if recall == 0 and precision == 0:
            f_measure = 0.
        else:
            f_measure = 2 * recall * precision / (recall + precision)
        total_recall += recall
        total_precision += precision
        total_fmeasure += f_measure
        metrics[label] = (recall, precision, f_measure)
        print('Recall for class {}: {}'.format(label, recall))
        print('Precision for class {}: {}'.format(label, precision))
        print('F-measure for class {}: {}'.format(label, f_measure))
    total_recall /= len(gt_label_map)
    total_precision /= len(gt_label_map)
    total_fmeasure /= len(gt_label_map)
    print('Average Recall across {} classes: {}'.format(len(gt_label_map), total_recall))
    print('Average Precision across {} classes: {}'.format(len(gt_label_map), total_precision))
    print('Average F-Measure across {} classes: {}'.format(len(gt_label_map), total_fmeasure))

    print('Computing Confusion Matrix')
    classes = sorted(list(gt_label_map.keys()))
    cm, idx_label_map = get_confusion_matrix(confusion, classes)
    plot_confusion_matrix(cm, classes, output_img_path)


if __name__ == '__main__':
    try:
        eval_task3(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(e)
        print('Usage Guide: python metric3.py <ground_truth_folder> <result_folder> <confusion_matrix_path>')

import os
import sys
import json
import numpy as np
import editdistance

from unicodeit import replace

IOU_THRESHOLD = 0.5


def bbox_iou(bboxes1, bboxes2, return_intersections=False):
    x11, y11, x12, y12 = np.split(bboxes1[:, :4], 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2[:, :4], 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    unionArea = (boxAArea + np.transpose(boxBArea) - interArea)
    iou = interArea / unionArea
    if return_intersections:
        sigmas = interArea / boxAArea
        taus = interArea / np.transpose(boxBArea)
        return iou, sigmas, taus
    else:
        return iou


def sanitize_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.lower().strip()
    if '\\' in text or '_{' in text or '^{' in text:
        text = replace([text])[0]
    return text


def extract_bboxes(js):
    text_blocks = js['task2']['output']['text_blocks']
    bboxes = []
    ids = []
    texts = []
    for text_block in text_blocks:
        bbox = text_block['bb']
        x1, y1, h, w = bbox['x0'], bbox['y0'], bbox['height'], bbox['width']
        x2, y2 = x1 + w, y1 + h
        if 'id' in text_block:
            ids += [text_block['id']]
        raw_text = text_block['text']
        text = sanitize_text(raw_text)
        if '__' in text:
            continue
        texts += [text]
        bboxes += [(x1, y1, x2, y2)]
    bboxes = np.asarray(bboxes)
    return bboxes, ids, texts


def eval_task2(gt_folder, result_folder):
    total_iou_score = 0.
    total_text_score = 0.
    for gt_file in os.listdir(gt_folder):
        gt_id = ''.join(gt_file.split('.')[:-1])
        # if this image has not been processed at all by a submission, it counts as zero for IOU and OCR scores
        if not os.path.isfile(os.path.join(result_folder, gt_id + '.json')):
            continue
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)
        gt_bboxes, gt_ids, gt_texts = extract_bboxes(gt)
        with open(os.path.join(result_folder, gt_id + '.json'), 'r') as f:
            res = json.load(f)
        res_bboxes, res_ids, res_texts = extract_bboxes(res)
        iou = bbox_iou(gt_bboxes, res_bboxes)
        iou_flag = iou >= IOU_THRESHOLD
        # fp_count = len(res_bboxes)
        # fn_count = len(gt_bboxes)
        iou_score = 0.
        text_score = 0.
        for g in range(len(gt_bboxes)):
            # exact match or one-many match
            if iou_flag[g, :].sum() >= 1:
                # take the best match in case of multiple predictions mapping to one gt
                # rest are considered FP
                r = np.argmax(iou_flag[g, :])
                iou_score += iou[g, r]
                ncer = editdistance.eval(gt_texts[g], res_texts[r]) / float(len(gt_texts[g]))
                text_score += max(1. - ncer, 0.)
                # fp_count -= 1.
                # fn_count -= 1.
        iou_score /= max(len(gt_bboxes), len(res_bboxes))
        text_score /= len(gt_bboxes)
        total_iou_score += iou_score
        total_text_score += text_score
    total_iou_score /= len(os.listdir(gt_folder))
    total_text_score /= len(os.listdir(gt_folder))
    hmean_score = 2 * total_iou_score * total_text_score / (total_iou_score + total_text_score)
    print('Total IOU Score over all ground truth images: {}'.format(total_iou_score))
    print('Total OCR Score over all ground truth images: {}'.format(total_text_score))
    print('Harmonic Mean of overall IOU and OCR scores: {}'.format(hmean_score))


if __name__ == '__main__':
    try:
        eval_task2(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(e)
        print('Usage Guide: python eval_task2.py <ground_truth_folder> <result_folder>')



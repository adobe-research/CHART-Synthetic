
import sys
import cv2
import json

if len(sys.argv) < 4:
    print("Usage: python visualize_json.py json_annotation input_im output_im")
    exit()

def find_by_id(_id, objs):
    for obj in objs:
        if obj['id'] == _id:
            return obj
    return None

in_json_file = sys.argv[1]
in_im_file = sys.argv[2]
out_file = sys.argv[3]

in_obj = json.load(open(in_json_file))
im = cv2.imread(in_im_file, 1)
h,w,_ = im.shape

# role_colors is also used to color lineplot lines
role_colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
           (255, 128, 0)] 
patch_color = (255, 0, 128)
patch_connect = (255, 128, 0)
element_colors = [(0, 255, 128), (128, 255, 0), (0, 128, 255), (128, 0, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128) ]
text_role_mapping = {'chart_title': 0, 'axis_title': 1, 'tick_label': 2, 'legend_title': 3, 'legend_label': 4}
boxplot_mapping = {'median': 0, 'min': 1, 'max': 2, 'first_quartile': 3, 'third_quartile': 4}

# task 1
chart_type = in_obj['task1']['output']['chart_type']
cv2.putText(im, chart_type, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2, bottomLeftOrigin=False)

if chart_type in ['Pie', 'Donut']:
    cv2.imwrite(out_file, im)
    exit()

# task 2 & 3
for text_block in in_obj['task2']['output']['text_blocks']:
    bb = text_block['bb']
    _id = text_block['id']
    obj = find_by_id(_id, in_obj['task3']['output']['text_roles'])
    assert obj is not None

    role = obj['role']
    color = role_colors[text_role_mapping[role]]
    p1 = (int(bb['x0']), int(bb['y0']))
    p2 = (int(bb['x0'] + bb['width']), int(bb['y0'] + bb['height']))
    cv2.rectangle(im, p1, p2, color, thickness=2)
    #if role == 'axis_title':
    cv2.putText(im, text_block['text'], p2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=2, bottomLeftOrigin=False)

# task 4
bb = in_obj['task4']['output']['_plot_bb']
p1 = (int(bb['x0']), int(bb['y0']))
p2 = (int(bb['x0'] + bb['width']), int(bb['y0'] + bb['height']))
cv2.rectangle(im, p1, p2, (0, 255, 0), thickness=2)

for axis, color in [('x', (255, 0, 0)), ('y', (255, 0, 255))]:
    for tick_obj in in_obj['task4']['output']['axes']['%s-axis' % axis]:
        _id = tick_obj['id']
        pt = tick_obj['tick_pt']
        x = pt['x']
        y = pt['y']
        cv2.circle(im, (x, y), 2, color, thickness=-1)

        label_bb = find_by_id(_id, in_obj['task5']['input']['task2_output']['text_blocks'])['bb']
        p = (int(label_bb['x0']), int(label_bb['y0']))
        cv2.line(im, (x,y), p, color, thickness=1)

# task 5
for obj in in_obj['task5']['output']['legend_pairs']:
    patch_bb = obj['bb']
    p1 = (int(patch_bb['x0']), int(patch_bb['y0']))
    p2 = (int(patch_bb['x0'] + patch_bb['width']), int(patch_bb['y0'] + patch_bb['height']))
    cv2.rectangle(im, p1, p2, patch_color, thickness=2)

    _id = obj['id']
    label_bb = find_by_id(_id, in_obj['task5']['input']['task2_output']['text_blocks'])['bb']
    p = (int(label_bb['x0']), int(label_bb['y0']))

    cv2.line(im, p1, p, patch_connect, thickness=1)

# task 6
idx = 0
for bb in in_obj['task6']['output']['visual elements']['bars']:
    p1 = (int(bb['x0']), int(bb['y0']))
    p2 = (int(bb['x0'] + bb['width']), int(bb['y0'] + bb['height']))
    color = element_colors[idx % len(element_colors)]
    cv2.rectangle(im, p1, p2, color, thickness=2)
    idx += 1

for boxplot in in_obj['task6']['output']['visual elements']['boxplots']:
    for name, obj in boxplot.items():
        color = element_colors[boxplot_mapping[name]]
        x = obj['x']
        y = obj['y']
        bb = obj['_bb']
        p1 = (int(bb['x0']), int(bb['y0']))
        p2 = (int(bb['x0'] + bb['width']), int(bb['y0'] + bb['height']))
        cv2.rectangle(im, p1, p2, color, thickness=1)
        cv2.circle(im, (x, y), 3, color, thickness=-1)

idx = 0
for line in in_obj['task6']['output']['visual elements']['lines']:
    for pt in line:
        x = pt['x']
        y = pt['y']
        color = role_colors[idx % len(role_colors)]
        cv2.circle(im, (x, y), 2, color, thickness=-1)
    idx += 1

idx = 0
for pt in in_obj['task6']['output']['visual elements']['scatter points']:
    x = pt['x']
    y = pt['y']
    color = element_colors[idx % len(element_colors)]
    cv2.circle(im, (x, y), 2, color, thickness=-1)
    idx += 1

cv2.imwrite(out_file, im)


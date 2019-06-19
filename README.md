# CHART-Synthetic

This repo contains the Synthetic Dataset used in the ICDAR2019 Competition on HArvesting Raw Tables from Infographics (CHART-Infographics).  It also contains the metric scripts used to evaluate predictions.

See [here](https://chartinfo.github.io/) for the original website describing the competition.

## Dependencies

numpy
matplotlib
scipy
editdistance
cv2 (visualize_json.py)
[unicodeit](https://github.com/svenkreiss/unicodeit/blob/master/src/unicodeit.py) (metric2.py only; just grab the file and get it on your $PYTHON_PATH)

## Tasks

The goal of extracting the raw data (e.g. a CSV file) from a chart image can be broken into several tasks that can be arranged in a pipeline fashion as shown below.

![Task Pipeline](task_pipeline.png)

In designing the competition, we wanted to analyze the errors made at each step independent of the errors made on previous tasks.  Hence, we provided the ideal outputs of all previous tasks as the input to each task.  We also wanted to evaluate the full end-to-end pipeline and provided a task for that as well.

Below each task is described:

### Task 1 - Chart Classification

Knowing the type of chart greatly affects what processing needs to be done. Thus, the first task is to classify chart images by type. Given the chart image, methods are expected to output one of the following 10 classes:

* Pie	
* Donut	
* Vertical box	
* Horizontal box
* Grouped vertical bar	
* Grouped horizontal bar	
* Stacked vertical bar	
* Stacked horizontal bar
* Line	
* Scatter

For bar charts that have a single set of bars (single data series), it is visually ambigious whether it a Grouped or Stacked bar chart, though their ground truth class indicates one of these classes. In this case we accept either prediction, as long as the orientation (vertical or horizontal) is correct. For example, a vertical bar chart with a single data series can be classified as either Grouped vertical bar or Stacked vertical bar.

Note that pie and donut plots are not used for the remaining tasks.

#### Metric
The evaluation metric will be the average per-class F-measure. Based on the class confusion matrix, we can compute the precision, recall, and F-measure for each class. The overall score is the average of each classes' F-measure.

To account for charts with multiple possible labels (i.e. single data series bar charts), the per-class precision and recall is modified to not penalize ambiguous cases.

#### Input/Output
Input: Chart Image

Output: Chart Class

### Task 2 - Text Detection and Recognition

Understanding the text in charts is necessary to interpret the graphical elements correctly. This task concentrates on detecting and recognizing the text within the chart image. Competing systems are expected to produce tight bounding boxes and transcriptions for each text block. Examples of individual text blocks individual titles, tick labels, legend labels. Text blocks may be a single line, multiple lines (due to text wrapping), and may be horizontal, vertical, or rotated. A predicted bounding box matches a GT bounding box if their Intersection Over Union (IOU) is at least 0.5, and tighter IOU criteria will be used to resolve ties when multiple predictions can match a single GT bounding box.

#### Metric
There are two evalaution metrics for detection and recognition respectively. For detection, we will sum the per-block IOU and divide by max(#predicted, #GT) for each image. For recognition, we will average normalized Character Error Rate (CER) for each text block in an image. By normalized CER, we mean that the number of character edits to transform a predicted word to GT word is divided by the length of the GT block. False positive and false negative text block detections will be assigned a normalized CER of 1 and an IOU of 0. We will use the same procedure as the ICDAR Robust Reading Competitions to handle split/merged boxes.

For each chart, we will compute both detection and recognition scores. Then we will average the per-chart scores over the whole dataset to ensure that each image contributes equally to the final score. The single summarizing metric is the harmonic mean of the aggregate detection and recognition scores.

#### Input/Output
Input: Chart Image, Chart Class

Output: List of (Text Block BBs, Text Transcription)

### Task 3 - Text Role Classification
For text to be useful in chart interpretation, its semantic role should be identified. This task focuses on identifying the role of each text block in a chart image, and text bounding boxes and transcripts are provided as input. Competing systems are expected to classify each bounding box into one of the following roles:

* Chart title	
* Axis title 
* Tick Label	
* Legend label


#### Metric
Similar to the evaluation in task 1 (chart classification), the evaluation metric will be the average per-class F-measure.

#### Input/Output
Input: Chart Image, Chart Class, List of (Text Block BB, Text Transcription, Text Block Id)

Output: List of (Text Block Id, Text role)

The output Text Block Ids should match the input Ids.

### Task 4 - Axis Analysis
Locating and interpreting the axes of the chart is critical to transforming data point coordinates from units of pixels to the semantic units. Competing systems are expected to output the location and value of each tick mark on both the X-axis and Y-axis. Tick locations are represented as points and must be associated with the corresponding value (a string). Note that some sets of ticks are ordered or unordered discrete sets with textual non-numeric labels.

For this dataset, X-axis will always refer to the axis that represents the independent variable shown, rather than the axis that is visually horizontal. For example, vertical bar and vertical box plots have an X-axis that is vertical. Similarly, the Y-axis is not always the axis that is vertical.

#### Metric
We use a modified F-measure to score each axis and then take the average F-measure over all axes. Each detected tick is scored for correctness, receiving a score between 0 and 1. Precision is then computed as the sum of the scores divided by the number of predictions. Recall is computed as the sum of the scores divided by the number of ground truth ticks.

A detected tick receives a score of 1 if the predicted point is close to the corresponding GT tick point, where correspondance between predictioned and GT ticks is based on the text BB and transcription. The threshold for close (scoring 1) and the threshold for far (scoring 0) is based on the distance between tick marks in the chart image. Predictions that are between the close and far thresholds are penalized linearly with distance.

#### Input/Output
Input: Chart Image, Chart Class, List of (Text Block BB, Text Transcription, Text Block Id)

Output: For each of X-axis and Y-axis, List of tuples (tick x position, tick y position, Text Block Id)

### Task 5 - Legend Analysis
The purpose of chart legends is to associate a data series name with the graphical style used to represent it. This is critical to chart understanding when there are multiple data series represented.

Competing systems are expected to associate each legend label text with the corresponding graphical style element within the legend area. Bounding boxes and transcriptions (but not text roles) are given as input. Note that in this task, legend labels are not paired with the corresponding data series found in the plot area. Also, some charts do not have legends, and an empty list should be returned.

#### Metric
For each GT legend label, if there is an associated predicted graphical style element, we compute the IOU of the predicted BB to the GT graphical style element BB. We then divide the sum of the IOU by max(#predicted, #GT) for each image, and then average this value over all images.

For charts that have no legend, it is expected that participant systems return an empty list to receive the max score for that chart. When there is no legened, specifying any output results in a score of 0 for that chart.

#### Input/Output
Input: Chart Image, Chart Class, List of (Text Block BB, Text Transcription, Text Block Id)

Output: A list of (Text Block Id, Graphical Style Element BB)


### Task 6 - Data Extraction
The goal of this task is to convert all of the previously extracted information into data series, which we define as sequences of (x,y) points. We break this task into 2 subtasks: (a) plot element detection and classification (b) data conversion. 

### Task 6a - Plot Element Detection/Classification
For 6a, the task of visual analysis, the goal is to detect and classify each individual element in the plot area. The representation of the element varies by class and is listed in the table below. Note that the output representations (BB or point) are in units of pixels.

Element Class |	Description |	Representation
--------------|-------------|---------------
Bar           |	Individual bars in bar charts |	Bounding Box
Line 	  |Location of Data Points in line charts	| Point Sequence
Scatter Marker|	Location of Data Points in scatter charts |	Point
Boxplot Median|	Median Line of Boxplot	| Point
Boxplot Box Top |	Line that is typically the upper quartile |	Point
Boxplot Box Bottom |	Line that is typically the lower quartile |	Point
Boxplot Top Wisker |	Line that is typically the max value |	Point
Boxplot Bottom Wisker |	Line that is typically the min value |	Point

Even though boxplot elements are visually line segments, we allow for any point on that line segment. Other plot elements, such as boxplot outlier points and error bars, are not evaluated and should not be contained in the output for this task. Note that the chart class is given as input to this task and that each plot element can be found in only one class of chart.

#### Metric
For an element to be correctly detected, it must be assigned to the correct class. We will use a variation on MSE to evaluate the representation of each element with the correct class. For each element, we compute a score between 0 and 1, where 1 represents an exact prediction, and predictions farther away than a distance threshold, T, receive a score of 0. The score is max(0, 1 - (D/T)), where D is the Manhattan distance between the predicted and GT points. The distance threshold, T, is determined to be 5% of the smallest image dimension. Because there are many ways to pair predicted and GT points, we will find the minimum cost pairing (i.e. solve this bi-partite graph matching problem).

For Boxplot elements, we will use distance between the predicted point and the line segment. For Bar chart bars, we will use the distances between corresponding BB corners.

For each chart, the scores will be summed and divided by max(#GT, #Predictions). Then these scores will be averaged across all images.

For line plots, individual lines must be segmented from each other, and are scored similarly as lines in 6b, except the units of predicted values should be in pixels for this task.

#### Input/Output
Input: Outputs of tasks 1-5

Output: List of (Element Class, Element Representation)

### Task 6b - Raw Data Extraction

Output the raw data that was used to generate the chart image. For the purpose of this competition, we define a simple schema, where each chart is a set of data series, and a data series is a name (string) and a list of (x,y) points. The x values can be either numerical or string values, depending on the X-axis domain. The y values are always numerical.

For box plots, it is not necessary to reproduce the raw data as the plot only shows a statistical summary. Instead, participants are expected to recover the dataset median, upper and lower quartiles, and wisker values. The interpretation of the wiskers (e.g. dataset min/max or 2/98 percentiles) is not always contained in the chart image itself, so we do not require this information at any stage of the competition.

#### Metric
Data Series names should come from the chart legend (if there is one). If the data series names are not specified in the chart image, then the predicted names are ignored for evaluation purposes.

See this [PDF](metric6b.pdf) for details

#### Input/Output
Input: Outputs of tasks 1-5.

Output: Set of Data Series. Data Series = (name, \[(x_1, y_1), ..., (x_n, y_n)])

The output of 6a is not given as an input to 6b.

## Task 7 - End-to-end Data Extraction
Perform task 6b using only the chart image as input.

# Repo Details

### Licensing

This work (code and data) is distributed with CC-BY-NC-ND 4.0 licensing terms.  See [license.txt](license.txt) for more details.

### Contributing

While this repository is not actively developed, if there are errors in the scripts, feel free to contribute (or open issues). Read the [Contributing Guide](./.github/CONTRIBUTING.md) for more information.

### Acknowledgements

This competition was jointly organized with the Center for Unified Biometrics and Sensors (CUBs), University at Buffalo, SUNY.  We thank Bhargava Urala and Kenny Davila for writing some of the metric scripts and for their permission to host them as part of this repo.


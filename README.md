# Computer Vision MMP Course

This repository contains solved laboratory assignments from the **Computer Vision** course, completed during the 3rd year of the **MMP program at CMC MSU**.

The course focuses on practical computer vision problems: image segmentation, object boundary detection, color classification, tile recognition and route reconstruction from visual data.

---

## Repository Overview

The repository consists of two laboratory assignments. Each lab is implemented in Python and includes source code, reports, datasets or additional assets used for experiments and result analysis.

```text id="ktljp6"
CV_MMP_course/
│
├── Lab1/
│   ├── Lab1_train/
│   ├── Lab1.py
│   ├── Lab1_2.py
│   ├── app.py
│   ├── app2.py
│   └── report.pdf
│
└── Lab2/
    ├── report_assets/
    ├── Образцы/
    ├── app.py
    ├── debug_pipeline.py
    ├── image_io.py
    ├── line_classification.py
    ├── report.pdf
    ├── report.tex
    ├── route_analysis.py
    ├── segmentation_tiles.py
    ├── tile_identification.py
    ├── tile_line_masks.py
    └── tile_normalization.py
```

---

## Topics Covered

### 1. Object Segmentation and Tomato Color Classification

The first laboratory assignment is dedicated to image segmentation and object analysis in photographs containing eggs and tomatoes.

This lab covers:

* separating eggs from tomatoes in an image;
* detecting object boundaries;
* extracting contours;
* analyzing object shapes;
* classifying tomatoes by color;
* processing real-world images with non-trivial lighting and background conditions;
* visualizing segmentation and classification results.

The task combines several classical computer vision steps: preprocessing, segmentation, contour detection, feature extraction and final object classification.

---

### 2. Tile Boundary Detection, Recognition and Route Reconstruction

The second laboratory assignment focuses on recognizing game-like tiles and reconstructing a route from them.

This lab covers:

* detecting tile boundaries;
* segmenting individual tiles from an input image;
* normalizing tile images;
* extracting tile line masks;
* classifying lines on tiles;
* identifying tile numbers;
* analyzing connections between tiles;
* reconstructing the final route.

The solution is organized as a modular pipeline. Separate Python files are responsible for image input/output, tile segmentation, tile normalization, tile identification, line mask extraction, line classification and route analysis.

---

## Laboratory Structure

### Lab 1

The first lab contains the implementation for processing photographs with eggs and tomatoes.

The pipeline includes:

1. image preprocessing;
2. segmentation of objects;
3. boundary and contour detection;
4. separation of eggs and tomatoes;
5. color-based tomato classification;
6. result visualization;
7. report with explanation and experimental results.

This assignment demonstrates how classical image processing techniques can be used to solve an applied object recognition task without relying on deep neural networks.

---

### Lab 2

The second lab contains a more complex computer vision pipeline for tile recognition and route construction.

The pipeline includes:

1. reading and preparing input images;
2. detecting tile regions;
3. normalizing detected tiles;
4. extracting visual features from each tile;
5. detecting tile lines;
6. recognizing tile numbers;
7. determining how tiles are connected;
8. building the route from the recognized structure;
9. generating visual and analytical outputs.

The modular structure makes the solution easier to debug, test and improve. Each part of the pipeline is separated into its own Python module.

---

## Main Skills Practiced

The repository demonstrates practical computer vision skills:

* working with real images;
* image preprocessing;
* segmentation;
* thresholding and masking;
* contour and boundary detection;
* color-based classification;
* geometric normalization;
* object recognition;
* route reconstruction;
* modular computer vision pipeline design;
* report preparation and result analysis.

---

## Technologies Used

The repository uses the Python computer vision and scientific computing ecosystem:

* **Python**
* **NumPy**
* **OpenCV**
* **Matplotlib**
* **scikit-image**
* **LaTeX**

The assignments rely on classical computer vision methods and custom processing pipelines implemented in Python.

---

## Course Progression

The course starts with object segmentation and color classification on natural images. This first task introduces basic image processing operations and shows how visual objects can be separated, analyzed and classified.

The second task moves to a more structured recognition problem. It combines segmentation, geometric processing, tile identification and graph-like route reconstruction into a complete computer vision pipeline.

Together, the two labs demonstrate the path from low-level image processing to a full applied recognition system.

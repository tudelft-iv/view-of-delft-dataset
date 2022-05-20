# Examples and Demos

We aim to share our data in a format very close to the KITTI dataset to make it easy to use for researchers.
Thus, many open source tool designed for KITTI (e.g. data loaders, visualizations, detection libraries) should be directly applicable to the VoD dataset with no or minimal changes. 
<br>

---

Nevertheless, we provide a python based development kit and a handful of examples of its usage in the form of jupyter notebooks for your convenience. There are three jupyter notebooks, where each helps with a particular aspect of processing or visualizing the dataset.

## 1_frame_information - Introduction to the data
This example introduces the classes that help to load data from the dataset for a specific frame. The visualization of the data is also included to aid understanding. Below is the embedded verison of the notebook, with a link attached for fullscreen view. 

[Open example in full-screen](https://tudelft-iv.github.io/view-of-delft-dataset/docs/notebook_html/1_frame_information/1_frame_information.html)

[Link To the Jupyter Notebook](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/1_frame_information.ipynb)

<p align="center"><iframe src="https://tudelft-iv.github.io/view-of-delft-dataset/docs/notebook_html/1_frame_information/1_frame_information.html" height="600" width="100%"></iframe></p>

<br>

---

## 2_frame_transformations - Transformations and localization
We provide a package that helps the transformation between different coordinate frames with the development kit. This example shows how to load and use the transformations between these frames,e.g., sensors and world coordinates. Furthermore, it also directly demonstrates how to plot the vehicle's location on an aerial map. Similar to the first notebook, below is the embedded version of the notebook, with a link attached for a fullscreen view. 

[Open example in full-screen](https://tudelft-iv.github.io/view-of-delft-dataset/docs/notebook_html/2_frame_transformations/2_frame_transformations.html)

[Link To the Jupyter Notebook](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/2_frame_transformations.ipynb)

<p align="center"><iframe src="https://tudelft-iv.github.io/view-of-delft-dataset/docs/notebook_html/2_frame_transformations/2_frame_transformations.html" height="600" width="100%"></iframe></p>


<br>

---

## 3_2d_visualization - 2D Visualization
This example notebook shows how the development kit can be used to take a frame from the set, and visualize its image with its point clouds (radar and/or LiDAR), annotations projected and overlaid.

[Open example in full-screen](https://tudelft-iv.github.io/view-of-delft-dataset/docs/notebook_html/3_3d_visualization/3_2d_visualization.html)
[Link To the Jupyter Notebook](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/3_2d_visualization.ipynb)

<p align="center"><iframe src="https://tudelft-iv.github.io/view-of-delft-dataset/docs/notebook_html/3_3d_visualization/3_2d_visualization.html" height="600" width="100%"></iframe></p>

<br>

---

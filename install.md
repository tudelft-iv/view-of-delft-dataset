# Development Kit Installation

1. Clone the repository: `git@gitlab.tudelft.nl:intelligent-vehicles/view-of-delft-dataset.git`
2. Inside the root folder, use the `environment.yml` to create a new conda environment using: `conda env create -f environment.yml`
3. Activate the environment using: `conda activate view-of-delft-env`
4. In the same terminal windows using typing `jupyter notebook` will start the notebook server.
5. To get started with the dataset, follow the instuctions in `1_frame_information.ipynb`. 

In case the the interactive plots do not show up in the notebooks use: `jupyter nbextension install --py --sys-prefix k3d`

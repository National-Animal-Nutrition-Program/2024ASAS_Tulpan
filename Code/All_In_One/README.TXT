Install Anaconda from: https://docs.anaconda.com/anaconda/install/
Alternatively you can install Conda from: https://conda.io/projects/conda/en/latest/index.html

How to setup the Python environment using conda:

1. Create environment

  conda create --name regression  python==3.10  -y

or use this to install the dependent libraries from the requirements.txt file. You can skip step 3.

  conda create --name regression --file requirements.txt  python==3.10  -y

2. Activate the environment

  conda activate regression

3. Install packages

  conda install -c conda-forge  scikit-learn -y
  conda install -c conda-forge  statsmodels -y
  conda install -c conda-forge  joblib -y
  conda install -c conda-forge  seaborn -y
  conda install -c conda-forge  matplotlib -y

4. Execute code

  python3  regression_singlemeasurements_pipeline.py  -in  ../../Datasets/MarshallEtAl2023_selected_measurements.csv

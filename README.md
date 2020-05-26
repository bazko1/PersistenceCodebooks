# Persistence codebooks
Repository contains code and experiments based on following <a href="https://github.com/bziiuj/pcodebooks" target="_blank">repository</a>.
The purpouse of this repo is to reproduce results and simplify further research using python3 as tool.
## Installation
We manage packages via anaconda.
The easiest setup would be:
 - install <a href="https://docs.conda.io/projects/conda/en/latest/index.html" target="_blank">conda</a> 
 - clone repository
 - enter repository
 - `conda env create -f environment.yml`
 - `conda activate myenv`
 
 ## Running
 Run: `jupyter notebook`
 - Experiments and visualizations for different data types are kept in `experiments` folder.
 - .py files contain preprocesing tools and models

 ### Using precomputed models
 Some models are already precomputed and added to repo with best hyperparameters found using gridsearch.
 In such case models will be loaded and can be used for further needs.
 ### Recalculating experiments, adding custom models
 - To recalculate experiment clear .dill files under paths experiments/dataset/precomputed/cv, experiments/dataset/precomputed/grid.
 - Apply modifications to Model_Comparison.ipynb
 - Re-evaluate notebook

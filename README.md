# Recommender Systems Using Factorizing Personalized Markov Chains (FPMC)

### Course: DSC 210: Numerical Linear Algebra
### Instructor: Dr. Tsui-wei Wang

## Instructions:

- Ensure that the following libraries are installed in python 3 environment:
	- multiprocessing
	- joblib
	- PyTorch
	- argparse
	- matplotlib

- For FPMC model:
	- open DSC210_FPMC_Model.ipynb inside the 'FPMC' directory
	- run the notebook from top to bottom
	- referenced: [S. H. Hwang - FPMC](https://github.com/stathwang/FPMC)

- For SASRec model:
	- open DSC210_SASRec_Model.ipynb inside the 'SASRec' directory
	- run the notebook from top to bottom
	- original implementation: [Zan Huang](https://github.com/pmixer/SASRec.pytorch)
	- *Note: our trained models are included in the repo. If you wish to run the training yourself, uncomment the first cell and set the device parameter to your device and run the cell.*


## Results

### FPMC Model Results
The performance of the three SASRec models with hidden dimensions $d=32,64,128$ are shown: 
<p align='center'>
	<img src= ./assets/FPMC_results.png />
</p>
We found that the FPMC model has the best performance when the hidden dimension is 64. Since the Personalized Markov Chain for Sets approach utilized extremely sparse transition matrices, an overly complicated model may lead to over-fitting to the training data, resulting in worse performance.

### SASRec Model Results
The performance of the three SASRec models with hidden dimensions $d=32,64,128$ are shown: 
<p align='center'>
	<img src= ./assets/SASRec_results.png />
</p>
The superior performance of the model with `hidden_units = 32` for all evaluation metrics suggests that a smaller embedding size offers a better balance between model
complexity and generalization for the dataset. This outcome likely stems from the simplicity or sparsity of the dataset, where larger embeddings might introduce unnecessary complexity, leading to overfitting. 

### Comparison
Below are two visualizations, each corresponding to one of the evaluation metrics used in our analysis to compare the performance of the two models. The first plot corresponds to HR@10, while the second corresponds to NDCG@10.
<p align='center'>
	<img src= ./assets/hr_compare.png width=50%/><img src= ./assets/ndcg_compare.png width=50%/>
</p>
We observed that the SASRec model performs better in both metrics since it works better for sparse datasets, better generalizes to unseen data, and captures both short-term and long-term dependencies, while the FPMC model only captures the short-term dependencies.




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
...

### SASRec Model Results
The performance of the three SASRec models with hidden dimensions $d=32,64,128$ are shown: 
<p align='center'>
	<img src= ./assets/SASRec_results.png />
</p>
The superior performance of the model with `hidden_units = 32` for all evaluation metrics suggests that a smaller embedding size offers a better balance between model
complexity and generalization for the dataset. This outcome likely stems from the simplicity or sparsity of the dataset, where larger embeddings might introduce unnecessary complexity, leading to overfitting. 

### Comparison
<p align='center'>
	<img src= ./assets/hr_compare.png width=50%/><img src= ./assets/ndcg_compare.png width=50%/>
</p>





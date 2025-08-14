# TC-MTLR
Code for reproducing results from the paper "Temporally Consistent Survival Prediction for Non-Uniform Longitudinal Data" by Harrison Fah, Russell Greiner, Roger A. Dixon. 

This repository builds upon the code provided by Mariana Vargas Vieyra for their paper "Deep End-to-End Survival Analysis with Temporal Consistency"

## Steps for Running Experiments

1. Install the following python packages (experiments were run using python version 3.10.11):

	numpy==2.2.4
	jax==0.5.3
	chex==0.1.89
	optax==0.2.4
	tqdm==4.67.1
	pandas==2.2.3
	matplotlib==3.10.1
	torch==2.6.0
	scikit-learn==1.6.1

	
2. Follow the instructions in the data folder to download and preprocess the datasets used in the experiments.

3. To perform short/small dataset experiments, run the following line (changing the dataset name in the file):
		python main.py --config config/configs_{pbc/aids/smallrw}.yaml
Then to plot the results, run the line (note the slightly different dataset names):
		python small_results_plotter.py --path Results/{aids/pbc2/small_rw}/results.json --name {PBC2/AIDS/SmallRW}

4. To perform long/large dataset experiments, run the following line:
		python main_large.py --config config/configs_{lastfm/nasa/largerw}.yaml
Then to print the results, run the line:
		python large_results_plotter.py --path Results/{lastfm/nasa/large_rw}/results.json --name {LastFM/NASA/LargeRW}

## Citing the paper
To cite the paper, use the following information:
		To be filled in after submission



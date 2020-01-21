# CisRegModels
Scripts for building computational models of gene regulation with tensorflow

# Installation
We recommend using Anaconda and pip to install this.  

This module requires tensorflow (ideally with GPU support since execution time is orders of magnitude slower without GPUs), as well as several other standard python modules (that pip will install).

1. Install CisRegModels:

`pip install git+https://github.com/Carldeboer/CisRegModels`

2. That's it! It should work now (please let me know if you encounter problems)

# Testing the installation
1. Clone the git repo somewhere:

`git clone https://github.com/Carldeboer/CisRegModels`
2. Move into the main directory

`cd CisRegModels`
3. Run the test.

`./test.bat`

# Examples
See the contents of test.bat for examples for how to train models.
The outputs of the model are located in the `example/` folder. Model parameters are the same order as the motif filters with the first motif numbered 0 (i.e. i=0).  Negative numbers indicate the values of constants. 
Motifs are represented as Kd matrices, which are essentially a negated log-odds PWM with a uniform background. See `example/PKdMs/` for examples.

# Tips for training models
Here, specific examples come from the [test.bat](https://github.com/Carldeboer/CisRegModels/blob/master/test.bat) file included in the repository.
1. Watch the MSE: When training a model, it is important that the model converge.  In some cases, the default initial parameters are so far from the biological parameters that by the time the model has centred on the data, the parameters are so extreme that the gradients become 0 (and so the parameters become fixed at this extreme value).  Thus, it is sometimes necessary to initialize parameters to either a small value (e.g. -ia 0.01  and -ip 0.01, for activities and potentiations, respectively), or a prior (e.g. -ic example/allTF_minKds_polyA_and_FZF1.txt ). On GPRA data (with a scale of 0-17), getting stuck at an MSE of ~18 indicates that the parameters have gone to irrecoverable extremes.
2. Play with the learning rate:  The learning rate depends partly on the scale of your data and so picking the best learning rate is non-trivial (but see [here](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10) and [here](https://hackernoon.com/8-deep-learning-best-practices-i-learned-about-in-2017-700f32409512). Included are ways to cycle or decay the learning rate (-clr and -lred options).  I have so far not found these to be useful when using the Adam Optimizer, but they may be better with other optimizers.
3. Progressive learning of parameters: If you aim to have your parameters correspond to specific TFs (i.e. you load the motifs of known factors and desire that at the end the parameters learned represent those same factors), it is best to learn the parameters of the model progressively. The idea here is that changing the motifs before you know what those motifs are doing may result in the motif changing into a motif for a related factor (to better reflect the (perhaps randomly initialized) activity/potentiation parameters, e.g.).  That said, there are certain parameter combinations that should be learned together (e.g. activities and potentiations), where, if learned separately, the models may converge to a solution that is suboptimal, but very difficult to break out of when a new parameter is introduced.
4. Optimizers: Included are three optimizers: Adam, RMSProp, and Momentum.  So far I have had the most success with Adam, but combining the others with cyclical learning rates and/or exponential decay of learning rates might yield better results in some circumstances.

# Citation
Please cite: [de Boer CG, Vaishnav ED, Sadeh R, Abeyta EL, Friedman N, and Regev A. Deciphering gene-regulatory logic with 100 million synthetic promoters. Nature Biotechnology. 38, pages56–65(2020).](https://www.nature.com/articles/s41587-019-0315-8)


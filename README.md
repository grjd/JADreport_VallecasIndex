# Selecting the most important self-assessed features for predicting conversion to Mild Cognitive Impairment with Random Forest and Permutation-based methods

This repository contains the code and the supplementary material of the paper of the paper:

Jaime Gómez-Ramírez, Marina Ávila-Villanueva, Miguel Ángel Fernández-Blázquez, **"Selecting the most important self-assessed features for predicting conversion to Mild Cognitive Impairment with Random Forest and Permutation-based method"**.
**Abstract
*Alzheimer’s Disease (AD) is a complex, multifactorial and comorbid condition. The asymptomatic behavior in the early stages makes the identification of the disease onset particularly challenging.
Mild cognitive impairment (MCI) is an intermediary stage between the expected decline of normal aging and the pathological decline associated with dementia. The identification of risk factors for MCI is thus sorely needed. 
Self-reported personal information such as age, education, income level, sleep, diet, physical exercise, etc. are called to play a key role not only in the early identification of MCI but also in the design of personalized interventions and the promotion of patients empowerment. 
In this study we leverage on The Vallecas Project, a large longitudinal study on healthy aging in Spain, to identify the most important self-reported features for future conversion to MCI. Using machine learning (random forest) and permutation-based methods we select the set of most important self-reported variables for MCI conversion which includes among others, subjective cognitive decline, educational level, working experience, social life, and diet. Subjective cognitive decline stands as the most important feature for future conversion to MCI across different feature selection techniques.*

## Overview
This repository contains:

    All the python code used to generate the results and plots of the paper 
    Report files generated by the random forest classifier implemented in the paper

You can download all the files as a zip using the "Clone or Download" button on the main repository page, or you can click through the file listings to view the code directly in GitHub.

## Requirements

### Python Packages
The code relies on the following python3.X+ libs:
* numpy
* pandas
* sklearn
* matplotlib
* seaborn
* graphviz
* shap
* pdpbox

 ## Usage 
 
```python
import paper
paper.main() 
```
Select a seed for reproducibility `np.random.seed(42)`

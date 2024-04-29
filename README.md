# eXplainable Authorship Identification

## Abstract
**Authorship Identification (AId)** has three main tasks: 

- **Same-Authorship Verification (SAV)**: (binary) given two documents, do they share the same author?
- **Authorship Verification (AV)**: (binary) given a document and a candidate author, is the candidate the real author of the document?
- **Authorship Attribution (AA)**: (multiclass) given a document and a set of candidate authors, what candidate is the real author of the document?

While many efforts in Authorship Identification have focused on testing the accuracy of different learning algorithms, or on proposing new sets of features that these algorithms could exploit, or simply on applying known techniques to various case studies, little or no effort has been devoted to endowing these systems with the ability to generate explanations for their predictions.

This fact represents indeed a very important gap in the literature, and a hindrance to a more widespread adoption of these technologies: the ability to provide justifications for their own predictions is a very important property for machine-learned systems in general, and even more so when these systems are involved in significant decisions-making processes, such as deciding on the authorship of written documents, with all its legal and ethical implications.

On one hand, a domain expert who has devoted a sizeable intellectual effort to determining the authorship of a given document is unlikely to blindly trust the prediction of an automatic system, unless the possibility to examine the reasons of its prediction and/or the inner working of the system is provided. On the other hand, the knowledge regarding the process of an authorship system might inspire the domain expert with new possible working hypotheses that had not been considered before.

In this project, we carry out an in-depth analysis of the suitability of a set of well-known general-purpose XAI methods to the three main AId tasks. In particular, we explore the following XAI methods:

- **feature ranking** for SAV
- **transformer probing** for AA
- **factuals / counterfactuals selection** for AV.

Note that each XAI method can be easily applied to any other AId task. In the code, each task prompts the specific XAI method just as an example.
## The dataset
In this project, we employ one dataset named "MedLatin" (it can be downloaded [here](https://doi.org/10.5281/zenodo.4298503)).

We select only 5 authors from the whole dataset, and divide the resulting dataset into the training set (90%) and test set (10%).


## The learning methods
As a 'classic' Machine Learning method, we experiment with Support Vector Machine. We fine-tune the hyper-parameters via 3-fold cross-validation.

We also experiment with a RoBERTa-based transformer pre-trained for Latin tasks (see [here](https://huggingface.co/pstroe/roberta-base-latin-cased3)). We fine-tune the model for 5 epochs.


### Code 
The code is organized as follows in the `src` directory:

- `main.py`
- `xai`: directory with the code for the XAI methods (feature ranking: `feature_importance.py`, transformer probing: `probing.py`, factuals/counterfactual selection: `records.py`).
- `models`: directory with the code for the preprocessing of the dataset and for the creation of the task pairs (`preprocessing.py`), and for the two models we adopt (`linear.py` and `transformer`).
- `helpers.py`: various functions potentially useful for multiple projects.
- `train.py`: training process for the AId methods.
- `validation.py`: evaluation process for the AId methods.

### References

Mattia Setzu, Silvia Corbara, Anna Monreale, Alejandro Moreo, and Fabrizio Sebastiani. 2024. Explainable Authorship Identification in Cultural Heritage Applications. J. Comput. Cult. Herit. https://doi.org/10.1145/3654675

# Quickstart
## Installation
```shell
mkvirtualenv -p python3.9 xaid
pip install -r requirements.txt
```
Remember to download the dataset, and to update the dataset folder in the variable __DATASET_FOLDER in `main.py` if needed.
## Run
Example to run the AV experiment, using the SVM method.
```
python main.py run --dataset medlatin --algorithm svm --task av --positive_sampling_size 5000 --output "../output/svm_av" --logging_level debug --n_jobs 8
```
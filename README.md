# eXplainable Authorship Identification
(to be edited) 

## Abstract
**Authorship Identification (AId)** has three main tasks: 

- **Same-Authorship Verification (SAV)**: (binary) given two documents, do they share the same author?
- **Authorship Verification (AV)**: (binary) given a document and a candidate author, is the candidate the real author of the document?
- **Authorship Attribution (AA)**: (multiclass) given a document ans a set of candidate authors, what candidate is the real author of the document?

We offer a method to obtain explanations for the predictions in the SAV task, and, through them, also for the predictions in the AA and SAV tasks. We experiment with both 'classic' Machine Learning methods, employing Support Vector Machine and Logistic Regression, and a BERT-based neural network. Each classifier is trained purely to perform SAV classification, and we solve the AA and AV task by using the predicted probabilities for the author pairs.


## The dataset
The dataset can be selected with the parameter `dataset_name` in `main.py` (however, at the moment we only have one dataset, named "victoria").

We employ the Victorian era dataset (available [here](https://dataworks.iupui.edu/handle/11243/23)). In order to limit the size of the data to process, we randomly select 5 authors and collect all their texts; we divide the resulting dataset into the `trainval` (90%) and `test` (10%) set, and we further divide `trainval` into the `training` (90%) and `validation` (10%) sets, which remain constant for all the experiments. No further pre-processing is done.


## The tasks
The task can be selected with the parameter `task` in `main.py`.

For this projects, the creation of textual pairs is of primary importance. The creation of the pairs for the specific task is controlled by the function `make_task_pairs` in `process_data.py`.

For the SAV experiments, we create `n` positive pairs for each author (where a positive pair is made of two texts by the same author) and `m` negative pairs in total (where a negative pair is made of two texts by different authors). In our experiments, `n=1000` and `m=5000` for training, and `n=100` and `m=500` for validation and test. Each pair is labelled as `1` (same author) or `0` (different author).

For the AV and AA experiments, we employ a classifier trained for SAV in order to solve these tasks as well. Thus, we only need to make the respective test set: in order to do so, for each test sample, we create `k` pairs for each author (where a pair is made of the test sample and a text by the author); in our experiments, `k=10`. Each pair is then classified as for SAV, and the test sample is classified as:

- for AA: `Ax` (where `Ax` is the author whose pairs have obtained the highest mean proability for the `1` class)
- for AV: same as for AA, and finally `1` if `Ax` is the author of interest, `0` otherwise. 

In case the required number of pairs is less than the total number of pairs that can be created, we always randomly select the required number of pairs.

NB: for AV, we randomly select one single author as the author of interest.


## The learning methods
The learner can be selected with the parameter `learner_name` in `main.py`.

As 'classic' Machine Learning methods, we experiment with both Support Vector Machine (`svm`) and Logistic Regression (`lr`). We fine-tune the hyper-parameters via gridseach on the validation set; the best configuration is then re-trained on the union of the training and validation set. 

We also experiment with a BERT-based neural network (`bert`). We train the model for 5 epochs with evaluation on the validation set at each epoch, and then perform early stopping after 3 epochs without improvement (up to a maximum of 50 epochs). Afterward, we make a final training on the validation set for 5 epochs (without evaluation).


### Code 
The code is organized as follows in the `src` directory:

- `main.py`
- `ml_classification.py`: code for classification tasks with classic ML algorithms
- `nn_classification.py`: code for classification tasks with a BERT-based neural network
- `process_data.py`: functions to process the dataset and create the task pairs
- `utils.py`: various functions potentially useful for multiple projects 
 
 


### References


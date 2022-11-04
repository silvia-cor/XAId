# Quickstart
## Installation
```shell
mkvirtualenv -p python3.9 xaid
pip install -r requirements.txt
cd data/dataset/
# download Victoria dataset
wget "https://dataworks.iupui.edu/bitstream/handle/11243/23/Gungor_2018_VictorianAuthorAttribution_data-train.csv"
cd ../../
python -m spacy download en_core_web_sm
```

## Run
You can train the model both through command line and python interface.

**Command Line**
```shell
python cli_train.py --dataset victoria\
                    --algorithm lr\
                    --task sav\
                    --nr_authors 3\
                    --output output_run\
                    --positive_sampling_size 2500\
                    --negative_sampling_size 1.0\
                    --logging_level debug
                    --n_jobs -1
```
where `--dataset` selects the dataset to use,
`--lr` selects the algorithm to use (in this case Logistic Regression),
`--task` selects the task (one of Same-Author Verification (sav), Authorship Attribution (aa) or Authorship Verification (av)),
`--output` is used to name the output files,
`--positive_sampling_size` and `--negative_sampling_size` are used to regulate the positive/negative sampling per author.
 
**Python**
```python
from train import train

model, hyperparameters, validation = train("victoria", "lr", "sav", output="output_run", n_jobs=-1)
```
Both runs train the desired model and output several files: one for the trained model (`output.pickle`), one for the model's hyperparameters (`output.model.json`), one for the model's validation (`output.validation.json`), and one for the overall run configuration (`output.json`). 
For an in-depth review of each parameter run `help(train)`:
```python
train(dataset: 'str', algorithm: 'str', task: 'str', nr_authors: 'int' = 10, sampling_size: 'int | float' = 1.0, negative_sampling_size: 'int | float' = 1.0, chr_n_grams: 'int' = 3, hyperparameters: 'Optional[dict]' = None, output: 'Optional[str]' = None, seed: 'int' = 42, n_jobs: 'int' = 1, logging_level: 'str' = 'info') -> 'Tuple[object, dict, dict]'
    Train a model with the given `algorithm` to perform `task` on the given `dataset`.
    Args:
        dataset: The dataset, currently only "victoria" is supported.
        algorithm: One of "svm", "lr".
        task: One of "sav" (Same-Author Verification), "av" (Authorship Verification), and "aa" (Authorship Attribution)
        nr_authors: Number of authors to randomly select. Applies to AA tasks.
        sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        chr_n_grams: Character n-grams for model training. Defaults to 3.
        hyperparameters: Hyperparameter distributions for the model selection.
        output: Output file for configuration. The script generates a `output.cfg` (holding run configuration), a
                `output.pickle` holding the trained model, and a `output.results.json` holding validation results.
        seed: Random seed for the experiments. Defaults to 42.
        n_jobs: Parallelism degree, defaults to 1. Use -1 to use all available resources.
        logging_level: Logging level, defaults to "info".
    
    Returns:
        A triple (model, optimal hyperparameters, validation dictionary).
```

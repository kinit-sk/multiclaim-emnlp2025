# Multilingual vs Crosslingual Retrieval of Fact-Checked Claims: A Tale of Two Approaches
This repository contains the code and data for the paper titled *Multilingual vs Crosslingual Retrieval of Fact-Checked Claims: A Tale of Two Approaches* accepted to EMNLP 2025 (preprint available on [arXiv](https://arxiv.org/abs/2505.22118)).

It extends the [code](https://github.com/kinit-sk/multiclaim) released with the original [MultiClaim dataset](https://zenodo.org/records/7737983) and the associated [EMNLP 2023 paper](https://aclanthology.org/2023.emnlp-main.1027/) as well as some of its later [refactorings](https://github.com/kinit-sk/claim-retrieval).


## Install

Install libraries:

```
pip install -r requirements.txt
```

Install `faiss`:

```
mkdir faiss
cd faiss
wget https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python -m pip install faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## Data

In `datasets/multiclaim_extended/` there are the following files: `fact_check_splits.csv`, `post_splits.csv`, and `fact_check_post_mapping_splits.csv`. There are also generated negative samples in `datasets/multiclaim_extended/negatives`, namely `negatives-random.csv`, `negatives-similarity.csv`, and `negatives-topic.csv`.

To generate the final dataset to reproduce the paper's results, copy the [MultiClaim v2 dataset](https://doi.org/10.5281/zenodo.15413169) into `datasets/multiclaim_extended/multiclaim_v2`. Three files are expected: `fact_checks.csv`, `posts.csv`, `fact_check_post_mapping.csv`. Then run `prepare_dataset.ipynb` to produce final dataset files with splits.


## Training

This part has been adapted from the [original repository](https://github.com/kinit-sk/multiclaim). It can be run in the same way as in the original paper (which implemented a random selection of negative samples) or with one of the negative sample selection strategies.

To use the negative sample selection strategies, the code expects the csv files with pre-selected negative samples in `datasets/multiclaim_extended/negatives`. The csv files should be named as `negatives-[strategy].csv`, where `strategy` is one of `(random, similarity, topic)`.

You can run the following example code, which selects all training data and fine-tunes the selected model (`intfloat/multilingual-e5-large`) for 1 epoch with 5 negatives samples per each positive one selected with a `similarity`-based strategy and evaluates it on the English versions of the dev and test sets:

```python
from src.datasets import dataset_factory
from src.training.backend import Config
from src.training.train import train

train_dataset = dataset_factory(
    'multiclaim_extended',
    crosslingual=False,
    fact_check_language=None,
    language=None,
    post_language=None,
    fact_check_languages_exclude=None,
    languages_exclude=None,
    post_languages_exclude=None,
    split='train',
    version='original',
    filter_mapped_fact_checks=True,
    negatives='similarity').load()

dev_datasets = {
    'dev_en': dataset_factory(
        'multiclaim_extended',
        crosslingual=False,
        fact_check_language=None,
        language='en',
        post_language=None,
        fact_check_languages_exclude=None,
        languages_exclude=None,
        post_languages_exclude=None,
        split='dev',
        version='original',
        filter_mapped_fact_checks=True).load()
}

test_datasets = {
    'test_en': dataset_factory(
        'multiclaim_extended',
        crosslingual=False,
        fact_check_language=None,
        language='en',
        post_language=None,
        fact_check_languages_exclude=None,
        languages_exclude=None,
        post_languages_exclude=None,
        split='test',
        version='original',
        filter_mapped_fact_checks=True).load()
}

cfg = Config()
cfg.train.num_epochs = 1
cfg.train.device = 'gpu'
cfg.train.num_negatives = 5
cfg.model.name = 'intfloat/multilingual-e5-large'
cfg.model.pooling = 'mean'

train(cfg, train_dataset, dev_datasets, test_datasets)
```

### Running experiments

To run the fine-tuning experiments, first create a YAML config in `configs` based on the provided `config/example-config.yaml`. The config allows to:

* configure the training data
* configure the dev datasets (one or more)
* configure the test datasets (one or more)
* provide training parameters and specify the base model to fine-tune

Then run the training script:

```
python -m scripts.fine_tuning_experiments --config configs/[config_name].yaml
```

Or as a background process:

```
nohup python -m scripts.fine_tuning_experiments --config configs/[config_name].yaml >./logs/training.log 2>&1 &
```

### Other scripts

Besides the script for training (`scripts/fine_tuning_experiments.py`), the `scripts` folder also contains a script to generate negative samples: `scripts/negative_sampling.py`.


## Citation
If you use the code or data in this repository, please cite the following paper:

```
@misc{ramponi2025multilingualvscrosslingualretrieval,
      title={Multilingual vs Crosslingual Retrieval of Fact-Checked Claims: A Tale of Two Approaches}, 
      author={Alan Ramponi and Marco Rovera and Robert Moro and Sara Tonelli},
      year={2025},
      eprint={2505.22118},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22118}, 
}
```
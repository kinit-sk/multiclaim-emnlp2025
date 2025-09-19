from collections import defaultdict
import logging
from typing import Generator

import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from src.evaluation.metrics import standard_metrics
from src.evaluation.utils import find_fact_check_ids


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predicted_ranks(predicted_ids: np.array, desired_ids: np.array, default_rank: int = None):
    """
    Return sorted ranks of the `desired_ids` in the `predicted_ids` array.

    If `default_rank` is set, the final array will be padded with the value for all the ids that were not present in the `predicted_ids` array.
    """

    predicted_ranks = dict()

    for desired in desired_ids:

        try:
            # +1 so that the first item has rank 1, not 0
            rank = np.where(predicted_ids == desired)[0][0] + 1
        except IndexError:
            rank = default_rank

        if rank is not None:
            predicted_ranks[desired] = rank

    return predicted_ranks


def process_results(gen: Generator, default_rank: int = None, csv_path: str = None):
    """
    Take the results generated from `gen` and process them. By default, only calculate metrics, but dumping the results into a csv file is also supported
    via `csv_path` attribute. For `default_rank` see `predicted_ranks` function.
    """

    ranks = list()
    rows = list()

    for predicted_ids, desired_ids, post_id in gen:
        post_ranks = predicted_ranks(predicted_ids, desired_ids, default_rank)
        ranks.append(post_ranks.values())

        if csv_path:
            rows.append((post_id, post_ranks, predicted_ids[:100]))

    logger.info(f'{sum(len(query) for query in ranks)} ranks produced.')

    if csv_path:
        pd.DataFrame(rows, columns=['post_id', 'desired_fact_check_ranks',
                     'predicted_fact_check_ids']).to_csv(csv_path, index=False)

    return standard_metrics(ranks)


def evaluate_post_fact_check_pairs(gen, dataset):
    """
    Evaluate the performance of the model on the given data generator.
    """

    desired_fact_check_ids = defaultdict(lambda: list())
    for fact_check_id, post_id in dataset.fact_check_post_mapping:
        desired_fact_check_ids[post_id].append(fact_check_id)

    logging.info(f'{len(desired_fact_check_ids)} posts to evaluate.')

    for predicted_fact_check_ids, post_id in gen:
        yield predicted_fact_check_ids, desired_fact_check_ids[post_id], post_id


def evaluate_multiclaim(dataset, pipeline, csv_path=None, language=None):
    df = pd.DataFrame(columns=['post_id', 'fact_check_ids', 'generated_output'])
    post_ids = None
    
    if csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        post_ids = list(df['post_id'].values)

    # id_to_post = get_evaluation_subset(dataset, language=language, previous_ids=post_ids)
    id_to_post = dataset.id_to_post

    for post_id, post in tqdm(id_to_post.items()):

        output = pipeline(query=post)
        original_output = output.get('generated_text', '')
        if 'generated_text' not in output:
            documents = output.get('top_k', [])
        else:
            documents = output.get('documents', [])
        if len(documents) == 0 or isinstance(documents[0], int):
            fact_check_ids = documents
        else:
            fact_check_ids = find_fact_check_ids(
                dataset, documents, pipeline.retrieved_documents)

        if csv_path is not None:
            df = pd.concat([df, pd.DataFrame(
                [[post_id, fact_check_ids, original_output]],
                columns=['post_id', 'fact_check_ids', 'generated_output'])])
            df.to_csv(csv_path, index=False)

        yield np.array(fact_check_ids), post_id

from collections import defaultdict
import logging
from typing import Generator

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import standard_metrics


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
    Take the results generated from `gen` and process them. By default, only calculate metrics, but dumping the results into a csv file is also supported via `csv_path` attribute. For `default_rank` see `predicted_ranks` function.
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


def evaluate_multiclaim(dataset, pipeline, csv_path=None):
    post_ids = list(dataset.id_to_post.keys())
    posts = list(dataset.id_to_post.values())

    output = pipeline(query=posts)
    fact_check_ids, sims = output.get('top_k', (np.array([]), np.array([])))

    if csv_path is not None:
        results = {
            'post_id': post_ids,
            'fact_check_ids': fact_check_ids.tolist(),
            'similarity_scores': sims.tolist()
        }
        df = pd.DataFrame.from_dict(results)
        df.to_csv(csv_path, index=False)
    
    for _, row in tqdm(df.iterrows()):
        yield np.array(row['fact_check_ids']), row['post_id']


def evaluate_post_fact_check_pairs_batch(gen, dataset, batch):
    """
    Evaluate the performance of the model on the given data generator.
    """

    desired_fact_check_ids = defaultdict(lambda: list())
    batch_ids = set(batch['post_ids'])
    for fact_check_id, post_id in dataset.fact_check_post_mapping:
        if post_id in batch_ids:
            desired_fact_check_ids[post_id].append(fact_check_id)

    logging.info(f'{len(desired_fact_check_ids)} posts to evaluate.')

    for predicted_fact_check_ids, post_id in gen:
        yield predicted_fact_check_ids, desired_fact_check_ids[post_id], post_id


def evaluate_multiclaim_batch(batch, pipeline, csv_path=None):
    post_ids = batch['post_ids']
    posts = batch['posts']

    output = pipeline(query=posts)
    fact_check_ids, sims = output.get('top_k', (np.array([]), np.array([])))

    logging.info(f'Retrieved ({len(fact_check_ids.tolist())}, {len(sims.tolist())}) for {len(post_ids)} posts.')

    if csv_path is not None:
        results = {
            'post_id': post_ids,
            'fact_check_ids': fact_check_ids.tolist(),
            'similarity_scores': sims.tolist()
        }
        df = pd.DataFrame.from_dict(results)
        df.to_csv(csv_path, index=False)
    
    for _, row in tqdm(df.iterrows()):
        yield np.array(row['fact_check_ids']), row['post_id']


def compute_fc_to_fc_similarity(fact_checks, pipeline, csv_path=None):
    fc_ids = [fc_id for fc_id, _ in fact_checks]
    fc_claims = [fc_claim for  _, fc_claim in fact_checks]

    output = pipeline(query=fc_claims)
    result_ids, result_sims = output.get('top_k', (np.array([]), np.array([])))

    if csv_path is not None:
        results = {
            'fc_id': fc_ids,
            'fact_check_ids': result_ids.tolist(),
            'similarity_scores': result_sims.tolist()
        }
        df = pd.DataFrame.from_dict(results)
        df.to_csv(csv_path, index=False)

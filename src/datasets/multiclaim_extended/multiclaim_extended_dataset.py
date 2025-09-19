import ast
import logging
import os
import random
from os.path import join as join_path
from typing import Iterable, Optional

import pandas as pd
import numpy as np

from src.datasets.custom_types import Language
from src.datasets.dataset import Dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiClaimExtendedDataset(Dataset):
    """MultiClaim extended dataset

    Class for extended version of the MultiClaim dataset that can load different variants. It requires data from `data/datasets/multiclaim_extended`.

    Initialization Attributes:
        crosslingual: bool  If `True`, only crosslingual pairs (fact-check and post in different languages) are loaded.
        fact_check_fields: Iterable[str]  List of fields used to generate the final `str` representation for fact-checks. Supports `claim` and `title`.
        fact_check_language: Optional[Language]  If a `Language` is specified, only fact-checks with that language are selected. 
        language: Optional[Language]  If a `Language` is specified, only fact-checks and posts with that language are selected. 
        post_language: Optional[Language]  If a `Language` is specified, only posts with that language are selected.
        fact_check_languages_exclude: Optional[Iterable[Language]]  If a list of `Language`s is specified, fact-checks in those languages are filtered out. 
        languages_exclude: Optional[Iterable[Language]]  If a list of `Language`s is specified, fact-checks and posts in those language are filtered out. 
        post_languages_exclude: Optional[Iterable[Language]]  If a list of `Language`s is specified, posts in those languages are filtered out.
        split: `train`, `test` or `dev`. `None` means all the samples. 
        version: 'original' or 'english'. Language version of the dataset. Currently not implemented.
        relationship: Optional[Iterable[str]]  Defaults to ("backlink", "claimreview_schema"), but can include other types of relationships, e.g., "candidate:transitive". If None, all pairs (mappings) are loaded.

        Also check `Dataset` attributes.

        After `load` is called, following attributes are accesible:
            fact_check_post_mapping: list[tuple[int, int]]  List of Factcheck-Post id pairs.
            id_to_fact_check: dict[int, str]  Factcheck id -> Factcheck text
            id_to_post: dict[int, str]  Post id -> Post text
            id_to_negatives: dict[int, list[str]] Post id -> List of Factcheck texts of negative samples


    Methods:
        load: Loads the data from the csv files. Populates `id_to_documents`, `id_to_post` and `fact_check_post_mapping` attributes.
    """

    our_dataset_path = join_path('.','datasets', 'multiclaim_extended')
    csvs_loaded = False
    # LANG_THRESHOLD = 0.2

    def __init__(
        self,
        crosslingual: bool = False,
        fact_check_fields: Iterable[str] = ('claim', ),
        fact_check_language: Optional[Language] = None,
        language: Optional[Language] = None,
        post_language: Optional[Language] = None,
        fact_check_languages_exclude: Optional[Iterable[Language]] = None,
        languages_exclude: Optional[Iterable[Language]] = None,
        post_languages_exclude: Optional[Iterable[Language]] = None,
        split: Optional[str] = None,
        version: str = 'original',
        relationship: Optional[Iterable[str]] = ("backlink", "claimreview_schema"),
        filter_mapped_fact_checks: bool = False,
        negatives: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert all(field in ('claim', 'title') for field in fact_check_fields)
        assert split in (None, 'dev', 'test', 'train')
        assert version in ('english', 'original')
        assert negatives in (None, 'random', 'similarity', 'topic')

        self.crosslingual = crosslingual
        self.fact_check_fields = fact_check_fields
        self.fact_check_language = fact_check_language
        self.language = language
        self.post_language = post_language
        self.fact_check_languages_exclude = fact_check_languages_exclude
        self.languages_exclude = languages_exclude
        self.post_languages_exclude = post_languages_exclude
        self.split = split
        self.version = version
        self.relationship = relationship
        self.filter_mapped_fact_checks = filter_mapped_fact_checks
        self.negatives = negatives
        

    @classmethod
    def maybe_load_csvs(cls):
        """
        Load the csvs and store them as class variables. When individual objects are initialized, they can reuse the same
        pre-loaded dataframes without costly text parsing.

        `OurDataset.csvs_loaded` is a flag indicating whether the csvs are already loaded.
        """

        if cls.csvs_loaded:
            return

        posts_path = join_path(cls.our_dataset_path, 'posts.csv')
        fact_checks_path = join_path(cls.our_dataset_path, 'fact_checks.csv')
        fact_check_post_mapping_path = join_path(
            cls.our_dataset_path, 'fact_check_post_mapping.csv')

        for path in [posts_path, fact_checks_path, fact_check_post_mapping_path]:
            assert os.path.isfile(path)

        logger.info('Loading fact-checks.')
        cls.df_fact_checks = pd.read_csv(fact_checks_path).fillna(
            '').set_index('fact_check_id')
        logger.info(f'{len(cls.df_fact_checks)} loaded.')

        logger.info('Loading posts.')
        cls.df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
        logger.info(f'{len(cls.df_posts)} loaded.')

        logger.info('Loading fact-check-post mapping.')
        cls.df_fact_check_post_mapping = pd.read_csv(fact_check_post_mapping_path)
        logger.info(f'{len(cls.df_fact_check_post_mapping)} loaded.')

        cls.csvs_loaded = True

    def load_negatives(self):
        if self.negatives:
            negatives_path = join_path(self.our_dataset_path, 'negatives', f'negatives-{self.negatives}.csv')
            assert os.path.isfile(negatives_path)

            logger.info('Loading negative samples.')
            df_negatives = pd.read_csv(negatives_path).fillna('').set_index('post_id')
            logger.info(f'{len(df_negatives)} loaded.')
            
            return df_negatives
        else: 
            return None

    def load(self):
        self.maybe_load_csvs()

        df_posts = self.df_posts.copy()
        df_fact_checks = self.df_fact_checks.copy()
        df_fact_check_post_mapping = self.df_fact_check_post_mapping.copy()

        if self.split:
            logger.info(f'Filtering by split: {self.split}')
            split_post_ids = set(self.split_ids(df_posts))
            df_posts = df_posts[df_posts.index.isin(split_post_ids)]
            logger.info(f'Posts remaining: {len(df_posts)}')

            split_mapping_ids = set(self.split_ids(df_fact_check_post_mapping))
            df_fact_check_post_mapping = df_fact_check_post_mapping[df_fact_check_post_mapping.index.isin(split_mapping_ids)]
            logger.info(f'Fact-check-post mappings remaining: {len(df_fact_check_post_mapping)}')

            split_fc_ids = set(self.split_ids(df_fact_checks))
            df_fact_checks = df_fact_checks[df_fact_checks.index.isin(split_fc_ids)]
            logger.info(f'Fact-checks remaining: {len(df_fact_checks)}')
        
        def get_fact_check_language(row, field):
            if row[f"{field}_detected_language"]:
                return row[f"{field}_detected_language"]
            else:
                 np.nan

        # Filter fact-checks by the language detected in claim
        if self.language or self.fact_check_language:
            def fact_check_language_filter(row):
                if (len(self.fact_check_fields) == 1):
                    fc_lang = get_fact_check_language(row, self.fact_check_fields[0])
                    return fc_lang == self.language or fc_lang == self.fact_check_language
                else:
                    field0_lang = get_fact_check_language(row, self.fact_check_fields[0])
                    field1_lang = get_fact_check_language(row, self.fact_check_fields[1])

                    field0_match = field0_lang == self.language or field0_lang == self.fact_check_language
                    field1_match = field1_lang == self.language or field1_lang == self.fact_check_language
                    
                    return field0_match and field1_match
            
            df_fact_checks = df_fact_checks[df_fact_checks.apply(fact_check_language_filter, axis=1)]
            logger.info(
                f'Filtering fact-checks by language: {len(df_fact_checks)} fact-checks remaining.')

        # Filter posts by the language detected in the combined distribution
        if self.language or self.post_language:
            def post_language_filter(row):
                return (row["post_detected_language"] == self.language) or (row["post_detected_language"] == self.post_language)

            df_posts = df_posts[df_posts.apply(post_language_filter, axis=1)]
            logger.info(f'Filtering posts by language: {len(df_posts)} posts remaining.')

        # Exclude fact-checks in specified languages detected in claim
        if self.languages_exclude or self.fact_check_languages_exclude:
            def fact_check_language_exclude(row):
                if (len(self.fact_check_fields) == 1):
                    fc_lang = get_fact_check_language(row, self.fact_check_fields[0])
                    if fc_lang is None:
                        fc_lang = ''
                    return (not self.languages_exclude is None and not fc_lang in self.languages_exclude) or \
                        (not self.fact_check_languages_exclude is None and not fc_lang in self.fact_check_languages_exclude)
                else:
                    field0_lang = get_fact_check_language(row, self.fact_check_fields[0])
                    field1_lang = get_fact_check_language(row, self.fact_check_fields[1])

                    field0_match = (not self.languages_exclude is None and not field0_lang in self.languages_exclude) or \
                        (not self.fact_check_languages_exclude is None and not field0_lang in self.fact_check_languages_exclude)
                    field1_match = (not self.languages_exclude is None and not field1_lang in self.languages_exclude) or \
                        (not self.fact_check_languages_exclude is None and not field1_lang in self.fact_check_languages_exclude)
                    
                    return field0_match and field1_match
            
            df_fact_checks = df_fact_checks[df_fact_checks.apply(fact_check_language_exclude, axis=1)]
            logger.info(f'Filtering fact-checks by languages to exclude: {len(df_fact_checks)} fact-checks remaining.')

        # Exclude posts in specified languages detected in posts
        if self.languages_exclude or self.post_languages_exclude:
            def post_language_exclude(row):
                return (not self.languages_exclude is None and not row["post_detected_language"] in self.languages_exclude) or \
                (not self.post_languages_exclude is None and not row["post_detected_language"] in self.post_languages_exclude)

            df_posts = df_posts[df_posts.apply(post_language_exclude, axis=1)]
            logger.info(f'Filtering posts by languages to exclude: {len(df_posts)} posts remaining.')

        # Create mapping variable
        post_ids = set(df_posts.index)
        fact_check_ids = set(df_fact_checks.index)
        if self.relationship:
            fact_check_post_mapping = set(
                (fact_check_id, post_id)
                for fact_check_id, post_id, relationship, _ in df_fact_check_post_mapping.itertuples(index=False, name=None)
                if fact_check_id in fact_check_ids and post_id in post_ids and relationship in self.relationship
            )
        else:
            fact_check_post_mapping = set(
                (fact_check_id, post_id)
                for fact_check_id, post_id, _, _ in df_fact_check_post_mapping.itertuples(index=False, name=None)
                if fact_check_id in fact_check_ids and post_id in post_ids
            )
        logger.info(f'Mappings remaining: {len(fact_check_post_mapping)}.')

        # Leave only crosslingual samples
        if self.crosslingual:
            crosslingual_mapping = set()

            if (len(self.fact_check_fields) == 1):
                for fact_check_id, post_id in fact_check_post_mapping:
                    # Here, we assume that all fact-check claims have exactly one language assigned (via Google Translate)
                    # We will have to rework this part if we want to support multiple detected languages.
                    fact_check_language = get_fact_check_language(df_fact_checks.loc[fact_check_id], self.fact_check_fields[0])
                    post_language = df_posts.loc[post_id]["post_detected_language"]

                    if fact_check_language and post_language and fact_check_language != post_language:
                        crosslingual_mapping.add((fact_check_id, post_id))
            else:
                for fact_check_id, post_id in fact_check_post_mapping:
                    fc_language0 = get_fact_check_language(df_fact_checks.loc[fact_check_id], self.fact_check_fields[0])
                    fc_language1 = get_fact_check_language(df_fact_checks.loc[fact_check_id], self.fact_check_fields[1])
                    post_language = df_posts.loc[post_id]["post_detected_language"]

                    if fc_language0 and fc_language1 and post_language and fc_language0 != post_language and fc_language1 != post_language:
                        crosslingual_mapping.add((fact_check_id, post_id))

        
            fact_check_post_mapping = crosslingual_mapping
            logger.info(f'Crosslingual mappings remaining: {len(fact_check_post_mapping)}')

        # Filtering posts if any crosslingual or language filter were applied
        remaining_post_ids = set(
            post_id for _, post_id in fact_check_post_mapping)
        df_posts = df_posts[df_posts.index.isin(remaining_post_ids)]
        logger.info(f'Filtering posts.')
        logger.info(f'Posts remaining: {len(df_posts)}')

        # Filtering fact-checks to have only those that have at least one matching post
        if self.filter_mapped_fact_checks:
            remaining_fc_ids = set(
                fc_id for fc_id, _ in fact_check_post_mapping)
            df_fact_checks = df_fact_checks[df_fact_checks.index.isin(remaining_fc_ids)]
            logger.info(f'Filtering fact-checks.')
            logger.info(f'Fact-checks remaining: {len(df_fact_checks)}')

        # Create object attributes
        self.fact_check_post_mapping = list(fact_check_post_mapping)

        # Create final `str` representations
        if (len(self.fact_check_fields) == 1):
            if self.version == 'english':
                field = f'{self.fact_check_fields[0]}_en'
            else:
                field = self.fact_check_fields[0]
            
            self.id_to_fact_check = {
                fact_check_id: self.clean_text(claim)
                for fact_check_id, claim in zip(df_fact_checks.index, df_fact_checks[field])
            }
        else:
            if self.version == 'english':
                field0 = f'{self.fact_check_fields[0]}_en'
                field1 = f'{self.fact_check_fields[1]}_en'
            else:
                field0 = self.fact_check_fields[0]
                field1 = self.fact_check_fields[1]

            self.id_to_fact_check = {
                fact_check_id: self.clean_text(" ".join([text1, text2]))
                for fact_check_id, text1, text2 in zip(df_fact_checks.index, df_fact_checks[field0], df_fact_checks[field1])
            }
        
        if self.version == 'english':
            post_field = 'post_body_en'
        else:
            post_field = 'post_body'

        self.id_to_post = {
            post_id: self.clean_text(post_text)
            for post_id, post_text in zip(df_posts.index, df_posts[post_field])
        }

        self.id_to_negatives = {}
        if self.negatives:
            df_negatives = self.load_negatives()
            df_negatives['fc_id'] = df_negatives['fc_id'].apply(lambda x: [int(fc_id) for fc_id in x.split(';')])

            logger.info(f'Filtering negative samples.')
            for post_id in self.id_to_post.keys():
                self.id_to_negatives[post_id] = [
                    self.clean_text(self.df_fact_checks.loc[fact_check_id][self.fact_check_fields[0]])
                    for fact_check_id in df_negatives.loc[post_id]['fc_id'] 
                ]
            logger.info(f'Negative samples remaining: {len(self.id_to_negatives)}')
        
        return self

    def split_ids(self, df):
        return df[df['split'] == self.split].index
    
    def __getitem__(self, idx):
        fc_id, p_id = self.fact_check_post_mapping[idx]
        
        if self.negatives:
            return self.id_to_fact_check[fc_id], self.id_to_post[p_id], self.id_to_negatives[p_id]

        return self.id_to_fact_check[fc_id], self.id_to_post[p_id]

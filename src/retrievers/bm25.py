import logging
import os
import pandas as pd
import pyterrier as pt
import shutil
import string
import numpy as np
from typing import Any
from unidecode import unidecode
from collections import defaultdict

from src.datasets.dataset import Dataset
from src.retrievers.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25(Retriever):
    """
    A class to represent a BM25 retriever.
    
    Attributes:
        name (str): The name of the retriever.
        top_k (int): The number of top documents to return.
        use_unidecode (bool): Whether to use unidecode.
        dataset (Dataset): The dataset.
        
    """
    def __init__(self, name: str = 'bm25', top_k: int = 25, use_unidecode: bool = True, dataset: Dataset = None, **kwargs: Any):
        super().__init__(name, top_k, dataset=dataset)
        self.use_unidecode = use_unidecode

    def create_index(self) -> None:
        """
        Create the PyTerrier index for the BM25 retriever.
        """
        if not pt.started():
            pt.init()

        pt_index_path = os.path.join('.', 'cache', 'pyterrier_index')
        if os.path.isdir(pt_index_path):
            shutil.rmtree(pt_index_path)

        self.docs = pd.DataFrame.from_dict(
            self.dataset.get_fact_checks())
        self.docs.columns = ['docno', 'text']
        self.docs['docno'] = self.docs['docno'].astype(str)

        if self.use_unidecode:
            self.docs['text'] = self.docs['text'].apply(unidecode)

        logger.info('Creating PyTerrier index.')
        df_indexer = pt.DFIndexer(pt_index_path, verbose=True)
        _ = df_indexer.index(self.docs['text'], self.docs['docno'])
        self.index = pt.IndexFactory.of(
            os.path.join(pt_index_path, 'data.properties'))
        self.model = pt.BatchRetrieve(self.index, wmodel='BM25')
        logger.info('Index created.')

    def preprocess_query(self, query: str) -> str:
        if self.use_unidecode:
            query = unidecode(query)

        # Remove punctuation because of Terrier parser
        query = ''.join(
            ch for ch in query if ch not in string.punctuation).strip()

        # Hand empty text cases
        if not query:
            query = 'unk'
        
        return query

    def retrieve(self, query: str|list) -> tuple:
        """
        Retrieve documents based on the query.
        
        Args:
            query (str): The query to retrieve documents.
            
        Returns:
            tuple: The top-k texts and their ids.
        """
        if not hasattr(self, 'index'):
            self.create_index()
        
        if isinstance(query, list):
            query = [self.preprocess_query(q) for q in query]
            query_df = pd.DataFrame({'qid': list(range(len(query))), 'query': query})
            result = self.model.transform(query_df)

            retrieved_fc = defaultdict(list, result.groupby('qid')['docno'].apply(lambda x: list(x.astype(int))).to_dict())
            retrieved_scores = defaultdict(list, result.groupby('qid')['score'].apply(lambda x: list(x.astype(float))).to_dict())
            
            top_k = [retrieved_fc[qid] for qid in query_df['qid'].values]
            scores = [retrieved_scores[qid] for qid in query_df['qid'].values]

            if self.top_k:
                top_k = [res[:self.top_k] + [None] * (self.top_k - len(res[:self.top_k])) for res in top_k]
                scores = [res[:self.top_k] + [None] * (self.top_k - len(res[:self.top_k])) for res in scores]
            
            return [], (np.array(top_k), np.array(scores))
        else:
            query = self.preprocess_query(query)
            query_df = pd.DataFrame({'qid': [0], 'query': [query]})
            result = self.model.transform(query_df)
           
            if self.top_k is None:
                top_k = list(result['docno'].astype(int))
                scores = list(result['score'].astype(float))
            else:
                top_k = list(result['docno'].astype(int))[:self.top_k]
                scores = list(result['score'].astype(float))[:self.top_k]
        
            top_k_texts = [
                self.dataset.get_fact_check(fc_id)
                for fc_id in top_k
            ]

            return top_k_texts, top_k, scores

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Union
import yaml


def _normalize(d: Union[dict, list, str]) -> Union[dict, list, Any]:
    """
    Normalize all 'None' strings into None in all depths of the dictionary
    
    Args:
        d: dict, list or string to normalize
        
    Returns:
        dict, list or element with all 'None' strings converted into None
    """
    if isinstance(d, dict):
        return {k: _normalize(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_normalize(v) for v in d]
    elif d == 'None':
        return None
    else:
        return d


@dataclass
class Dataset:
    """
    name: identifier of the dataset
    crosslingual: whether we want to use a crosslingual pairs
    fact_check_language: language of the fact checking dataset
    fact_check_fields: fields to be used for fact checking
    language: language of the knowledge base
    post_language: language of the postprocessing
    split: which split of the dataset to use (train, dev, test)
    document_path: path to the document or directory with documents
    version: version of the dataset (original, english)
    """
    name: str = None
    crosslingual: bool = False
    fact_check_language: str = None
    fact_check_fields: Iterable[str] = ('claim', )
    language: str = None
    post_language: str = None
    split: str = None
    document_path: str = None
    version: str = 'original'  # original, english"


@dataclass
class Retriever:
    """
    name: identifier of the retriever
    model_name: name of the model to be used, e.g. 'intfloat/multilingual-e5-large'
    top_k: number of documents to retrieve
    use_unidecode: whether to use unidecode for the query, specific for bm25 retriever
    dataset: knowledge base to be used for the retrieval
    cache: path to the cache
    """
    name: str = None
    model_name: str = None
    top_k: int = 5
    use_unidecode: bool = False
    dataset: Dataset = field(default_factory=Dataset)
    cache: str = None


@dataclass
class Postprocessor:
    """
    name: identifier of the postprocessor
    """
    name: str = None


@dataclass
class PipelineConfig:
    """
    Config class to hold the configuration of the pipeline
    
    steps: list of steps in the pipeline
    
    """
    steps: List = None

    @classmethod
    def from_dict(cls, config: dict) -> 'PipelineConfig':
        """
        Create Config object from a dictionary
        
        Args:
            config: dictionary with the configuration of the pipeline
            
        Returns:
            PipelineConfig object
        """
        steps = []
        for step in config['steps']:
            step = _normalize(step)
            if 'retriever' in step.keys():
                dataset = Dataset(
                    **step['retriever']['dataset'])
                step = {k: v for k, v in step['retriever'].items(
                ) if k != 'dataset'}
                retriever = Retriever(**step, dataset=dataset)
                steps.append(retriever)
            elif 'postprocessor' in step.keys():
                postprocessor = Postprocessor(**step['postprocessor'])
                steps.append(postprocessor)

        return cls(steps=steps)

    @classmethod
    def load_config(cls, path: str) -> 'PipelineConfig':
        """
        Load configuration from a yaml file
        
        Args:
            path: path to the yaml file with the configuration
            
        Returns:
            PipelineConfig object
        """
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        
        return cls.from_dict(config)

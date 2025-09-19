from typing import Any, Union, List

from src.config import PipelineConfig, Retriever, Postprocessor
from src.retrievers import retriever_factory
from src.retrievers.retriever_postprocessor import postprocessor_factory
from src.datasets import dataset_factory
from src.datasets.dataset import Dataset
from src.datasets.retrieved_documents import RetrievedDocuments
from src.retrievers.retriever import Retriever as RetrieverModule

from copy import deepcopy


class Pipeline:
    """
    Pipeline class to hold the pipeline
    
    Args:
        path: path to the configuration file
        pipeline_config: configuration of the pipeline
    """
    def __init__(self, path: str = None, pipeline_config: dict = None):
        self.name = 'Pipeline'
        self.path = path
        self.config = pipeline_config
        self.steps = []
        self.load()
        
    def get_dataset(self, dataset: Any) -> Dataset:
        dataset = dataset.__dict__
        name = dataset.pop('name')

        if name == 'retrieved_documents':
            return None

        return dataset_factory(
            name, **dataset
        ).load()
        
    def get_pipeline_step(self, step: Union[Retriever, Postprocessor]) -> Union[Retriever, Postprocessor]:
        """
        Get the pipeline step based on the step type
        
        Args:
            step: configuration of the pipeline step
            
        Returns:
            Union[Retriever, Postprocessor]: the initialized pipeline step (a retriever or a postprocessor)
        """
        if isinstance(step, Retriever):
            cache = step.cache
            dataset = self.get_dataset(step.dataset)
            return retriever_factory(step.name, model_name=step.model_name, top_k=step.top_k, dataset=dataset, cache=cache)
        elif isinstance(step, Postprocessor):
            rest = {
                k: v for k, v in step.__dict__.items() if k not in ['name']
            }
            return postprocessor_factory(step.name, **rest)
        else:
            raise ValueError(f'Unknown step type: {type(step)}')

    def _convert_kwargs(self, step: Union[Retriever, Postprocessor], kwargs: dict) -> dict:
        if isinstance(step, RetrieverModule):
            self.retrieved_documents = kwargs['documents']
            self.retrieved_documents_ids = kwargs['top_k']

        return kwargs

    def __call__(self, **kwargs) -> Any:
        for idx, step in enumerate(self.steps):
            if isinstance(step, RetrieverModule) and isinstance(self.steps[idx - 1], RetrieverModule):
                if len(kwargs['documents']) == 0:
                    continue
                dataset = RetrievedDocuments(
                    name='retrieved_documents',
                    documents=kwargs['documents'],
                    ids=kwargs['top_k']
                )
                kwargs = {
                    "query": kwargs['query']
                }
                step.set_dataset(dataset)

            kwargs = step(**kwargs)
            kwargs = self._convert_kwargs(step, kwargs)

        return kwargs
    
    def _load_pipeline_steps(self) -> None:
        """
        Load the pipeline steps from the configuration
        """
        pipeline_steps = deepcopy(self.config.steps)

        self.steps = []
        for step in pipeline_steps:
            pipeline_step = self.get_pipeline_step(step)
            self.steps.append(pipeline_step)
            
    def set_steps(self, steps: List[Union[Retriever, Postprocessor]]) -> None:
        """
        Set the steps of the pipeline manually.
        
        Args:
            steps: list of steps
        """
        self.steps = steps

    def load(self) -> None:
        if self.config:
            self.config = PipelineConfig.from_dict(self.config)
            self._load_pipeline_steps()
        elif self.path:
            self.config = PipelineConfig.load_config(self.path)
            self._load_pipeline_steps()

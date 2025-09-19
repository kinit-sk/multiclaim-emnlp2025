def postprocessor_factory(name: str, **kwargs) -> 'RetrieverPostprocessor':
    """
    Factory function to create a postprocessor object.
    
    Args:
        name (str): The name of the postprocessor.
        
    Returns:
        Postprocessor: An instance of the postprocessor.
    """
    potsprocessor = {
        'retriever_postprocessor': RetrieverPostprocessor,
    }[name]
    return potsprocessor(**kwargs)


class RetrieverPostprocessor:
    def __init__(self, **kwargs):
        self.name = 'retriever_postprocessor'

    def postprocess(self, **kwargs) -> dict:
        """
        Postprocess the output of the retriever.

        Returns:
            dict: The postprocessed output and the documents.
        """
        return {
            'documents': kwargs['documents'],
            'top_k': kwargs['top_k']
        }
    
    def __call__(self, **kwargs):
        return self.postprocess(**kwargs)
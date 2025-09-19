import torch
from torch import nn
from sentence_transformers import util
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)


class MNRloss(nn.Module):
    logger = logging.getLogger('Loss')

    def __init__(self, label_smoothing=0, scale=20.0, similarity_f='cosine', as_hard_negatives=True):
        super().__init__()
        self.loss_f = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # If similarity function is set to dot product, then set scale to 1.0
        self.scale = scale
        if similarity_f == 'cosine':
            self.similarity_f = util.cos_sim
        else:
            self.similarity_f = None
        
        self.as_hard_negatives = as_hard_negatives

    def forward(self, sentence_embedding_A: torch.Tensor, sentence_embedding_B: torch.Tensor, 
                negative_embeddings: Optional[List[torch.tensor]]=None):
        # Compute similarity matrix
        if self.similarity_f:
            scores = self.similarity_f(sentence_embedding_B, sentence_embedding_A) * self.scale
        else:
            scores = torch.mm(sentence_embedding_A, sentence_embedding_B.transpose(0, 1))
        # Compute labels
        labels = torch.arange(len(scores), dtype=torch.long, device=scores.device)

        if negative_embeddings:
            neg_scores = [
                self.similarity_f(embedding, negative_embedding) * self.scale if self.similarity_f else torch.matmul(embedding, negative_embedding.transpose(0,1))
                for embedding, negative_embedding in zip(sentence_embedding_B, negative_embeddings)
            ]
            if self.similarity_f:
                neg_scores = [neg.squeeze(0) for neg in neg_scores]

            pos_scores = scores
            if self.as_hard_negatives:
                scores = torch.cat((pos_scores, torch.stack(neg_scores)), dim=1)
            else:
                scores = torch.stack([
                    torch.cat((score[label].unsqueeze(0), neg_score), 0)
                    for score, label, neg_score in zip(pos_scores, labels, neg_scores)
                ])
                labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)

        return self.loss_f(scores, labels)
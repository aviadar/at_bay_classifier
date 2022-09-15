from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from enum import Enum


class ModelType(Enum):
    facebook_bart_large_mnli = "facebook/bart-large-mnli"
    nli_distilroberta_base = "cross-encoder/nli-distilroberta-base"
    navteca_bart_large_mnli = "navteca/bart-large-mnli"


class TokenizerType(Enum):
    navteca_bart_large_mnli = "navteca/bart-large-mnli"


class TextClassifier:
    def __init__(self,
                 model_type: ModelType = ModelType.navteca_bart_large_mnli.value,
                 tokenizer_type: TokenizerType = TokenizerType.navteca_bart_large_mnli.value):
        model = AutoModelForSequenceClassification.from_pretrained(model_type)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.classifier = pipeline(task='zero-shot-classification',
                                   # device=0, # use this for GPU
                                   model=model,
                                   tokenizer=tokenizer)

    def classify(self, input_text: str, candidate_labels: list[str]) -> dict:
        res = self.classifier(input_text, candidate_labels, multi_label=False)
        return res

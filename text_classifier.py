from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from enum import Enum
from summarizer import Summarizer


class GpuUsage(Enum):
    On = 0
    Off = 1


class ModelType(Enum):
    facebook_bart_large_mnli = "facebook/bart-large-mnli"
    nli_distilroberta_base = "cross-encoder/nli-distilroberta-base"
    navteca_bart_large_mnli = "navteca/bart-large-mnli"


class TokenizerType(Enum):
    navteca_bart_large_mnli = "navteca/bart-large-mnli"


class TextClassifier:
    def __init__(self,
                 model_type: ModelType = ModelType.navteca_bart_large_mnli.value,
                 tokenizer_type: TokenizerType = TokenizerType.navteca_bart_large_mnli.value,
                 gpu: GpuUsage = GpuUsage.On):
        model = AutoModelForSequenceClassification.from_pretrained(model_type)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        if gpu == GpuUsage.On:
            self.classifier = pipeline(task='zero-shot-classification',
                                       device=0,  # use this for GPU
                                       model=model,
                                       tokenizer=tokenizer)
        else:
            self.classifier = pipeline(task='zero-shot-classification',
                                       model=model,
                                       tokenizer=tokenizer)

        self.summarizer = Summarizer(gpu.value)

    def classify(self, input_text: str, candidate_labels: list[str]) -> dict:
        reduced_txt = ' '.join(x.strip() for i, x in enumerate(input_text.split()) if i < 300)
        res = self.classifier(self.summarizer.summarize(reduced_txt), candidate_labels, multi_label=False)
        return res

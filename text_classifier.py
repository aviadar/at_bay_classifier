from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from enum import Enum
from summarizer import Summarizer
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


def get_lang_detector(nlp, name):
    return LanguageDetector()


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

        self.nlp = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=get_lang_detector)
        self.nlp.add_pipe('language_detector', last=True)

        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", model_max_len=1024)

    @staticmethod
    def divide_chunks(input_txt, n_chunks):
        for i in range(0, len(input_txt), n_chunks):
            yield input_txt[i:i + n_chunks]

    def classify(self, input_text: str, candidate_labels: list[str]) -> dict:
        reduced_txt = ' '.join(x.strip() for i, x in enumerate(input_text.split()) if i < 250)
        # txt_chunks = list(TextClassifier.divide_chunks([x.strip() for i, x in enumerate(input_text.split())], 200))
        # summary = ''
        # for chunk in txt_chunks:
        #     summary += ' ' + self.summarizer.summarize(chunk)

        lang = self.nlp(reduced_txt)._.language # 3
        if lang['language'] == 'en':
            res = self.classifier(self.summarizer.summarize(reduced_txt), candidate_labels, multi_label=False)
        else:
            res = None
        return res

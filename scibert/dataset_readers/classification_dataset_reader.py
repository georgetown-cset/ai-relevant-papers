""" Data reader for AllenNLP """

import logging
from typing import Dict, Any

import jsonlines
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField, MultiLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("classification_dataset_reader")
class ClassificationDatasetReader(DatasetReader):
    """
    Text classification data reader

    The data is assumed to be in jsonlines format
    each line is a json-dict with the following keys: 'text', 'label', 'metadata'
    'metadata' is optional and only used for passing metadata to the model
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 multilabel: bool = False,
                 ) -> None:
        super().__init__(lazy)
        self.multilabel = multilabel
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.multilabel = multilabel

    @overrides
    def _read(self, file_path):
        with jsonlines.open(file_path) as f_in:
            for json_object in f_in:
                yield self.text_to_instance(
                    text=json_object.get('text'),
                    label=json_object.get('label'),
                    metadata=json_object.get('meta')
                )

    @overrides
    def text_to_instance(self,
                         text: str,
                         label: str = None,
                         metadata: Any = None) -> Instance:  # type: ignore
        text_tokens = self._tokenizer.tokenize(text)
        fields = {
            'text': TextField(text_tokens, self._token_indexers),
        }
        if label is not None:
            if self.multilabel:
                # MultiLabelField expects either sequences of labels like ['cs_AI', 'cs_LG'], ['cs_AI'], ['cs_LG'], ... 
                # or alternatively 0-indexed integers
                # References:
                #   https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/multilabel_field.py
                #   https://github.com/allenai/allennlp/blob/master/allennlp/tests/data/fields/multilabel_field_test.py
                fields['label'] = MultiLabelField(label)
            else:
                fields['label'] = LabelField(label)

        if metadata:
            fields['metadata'] = MetadataField(metadata)
        return Instance(fields)

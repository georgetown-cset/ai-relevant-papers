from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('scibert_predictor')
class ScibertPredictor(Predictor):

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')
        instance.fields['text']

        output = {
            'prediction': {label: outputs['class_probs'][i] for i, label in label_vocab.items()},
            # TODO: this may assume InstanceLabel rather than MultiLabel
            'labels': instance.fields['label'].label,
            'metadata': instance.fields['metadata'].metadata,
            # TODO: would be nice to make passing the text through optional
            # Output space-delimited tokens, which may not be the input text but is useful for debugging
            'text': ' '.join([t.text for t in instance.fields['text'].tokens]),
        }
        return sanitize(output)

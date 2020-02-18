import ast
from typing import Dict, Optional, List, Any

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from pytorch_toolbelt.losses import BinaryFocalLoss, FocalLoss


@Model.register("text_classifier")
class TextClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 verbose_metrics: False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 loss: Optional[dict] = None,
                 ) -> None:
        super(TextClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward
        self.prediction_layer = torch.nn.Linear(self.classifier_feedforward.get_output_dim(), self.num_classes)
        self.pool = lambda text, mask: util.get_final_encoder_states(text, mask, bidirectional=True)

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}
        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)
        self.verbose_metrics = verbose_metrics

        if loss is None:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            alpha = loss.get('alpha')
            gamma = loss.get('gamma')
            weight = loss.get('weight')
            if alpha is not None:
                alpha = float(alpha)
            if gamma is not None:
                gamma = float(gamma)
            if weight is not None:
                weight = torch.tensor([1.0, float(weight)])
            if loss.get('type') == 'CrossEntropyLoss':
                self.loss = torch.nn.CrossEntropyLoss(weight=weight)
            elif loss.get('type') == 'BinaryFocalLoss':
                self.loss = BinaryFocalLoss(alpha=alpha, gamma=gamma)
            elif loss.get('type') == 'FocalLoss':
                self.loss = FocalLoss(alpha=alpha, gamma=gamma)
            elif loss.get('type') == 'MultiLabelMarginLoss':
                self.loss = torch.nn.MultiLabelMarginLoss()
            elif loss.get('type') == 'MultiLabelSoftMarginLoss':
                self.loss = torch.nn.MultiLabelSoftMarginLoss(weight)
            else:
                raise ValueError(f'Unexpected loss "{loss}"')

        initializer(self)

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:
        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self.text_field_embedder(text)

        mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, mask)
        pooled = self.pool(encoded_text, mask)
        ff_hidden = self.classifier_feedforward(pooled)
        logits = self.prediction_layer(ff_hidden)
        class_probs = F.softmax(logits, dim=1)

        output_dict = {"logits": logits}
        if label is not None:
            loss = self.loss(logits, label)
            output_dict["loss"] = loss

            # compute F1 per label
            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probs, label)
            self.label_accuracy(logits, label)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probs'] = class_probabilities
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            if self.verbose_metrics:
                metric_dict[name + '_P'] = metric_val[0]
                metric_dict[name + '_R'] = metric_val[1]
                metric_dict[name + '_F1'] = metric_val[2]
            sum_f1 += metric_val[2]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names)
        average_f1 = sum_f1 / total_len
        metric_dict['average_F1'] = average_f1
        metric_dict['accuracy'] = self.label_accuracy.get_metric(reset)
        return metric_dict

import ast
import logging
from collections import OrderedDict
from statistics import mean
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

from scibert.metrics.multilabel_f1 import MultiLabelF1Measure


@Model.register("multilabel_text_classifier")
class MultilabelTextClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 verbose_metrics: False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 loss: Optional[dict] = None,
                 ) -> None:
        super(MultilabelTextClassifier, self).__init__(vocab, regularizer)

        self.log = logging.getLogger(__name__)
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.log.warning(f'num_classes: {self.num_classes}')
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward
        self.log.warning(f'output_dim: {self.classifier_feedforward.get_output_dim()}')
        self.prediction_layer = torch.nn.Linear(self.classifier_feedforward.get_output_dim(), self.num_classes)
        self.pool = lambda text, mask: util.get_final_encoder_states(text, mask, bidirectional=True)

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = OrderedDict()
        self.verbose_metrics = verbose_metrics
        for i in range(self.num_classes):
            label = vocab.get_token_from_index(index=i, namespace="labels")
            self.log.warning(f'label {i}: {label}')
            self.label_f1_metrics[label] = F1Measure(positive_label=i)
        self.micro_f1 = MultiLabelF1Measure()
        self.label_f1 = OrderedDict()
        for i in range(self.num_classes):
            label = vocab.get_token_from_index(index=i, namespace="labels")
            self.label_f1[label] = MultiLabelF1Measure()

        if loss is not None:
            alpha = loss.get('alpha')
            gamma = loss.get('gamma')
            weight = loss.get('weight')
            if alpha is not None:
                alpha = float(alpha)
            if gamma is not None:
                gamma = float(gamma)
            if weight is not None:
                weight = torch.tensor(ast.literal_eval(weight))
        if loss is None or loss.get('type') == 'CrossEntropyLoss':
            self.loss = torch.nn.CrossEntropyLoss()
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
            From a ``LabelField`` or ``MultiLabelField``, a tensor of shape ``(batch_size, num_labels)``.
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

        hidden = self.classifier_feedforward(pooled)  # batch size x hidden size
        logits = self.prediction_layer(hidden)  # batch size x num labels

        # Reference: https://pytorch.org/docs/master/nn.html#sigmoid
        probabilities = torch.sigmoid(logits)  # batch size x num labels

        output_dict = {"logits": logits, "class_probs": probabilities}
        if label is not None:
            predictions = (logits.data > 0.0).long()
            label_data = label.squeeze(-1).data.long()
            self.micro_f1(predictions, label_data)
            output_dict["loss"] = self.loss(logits.squeeze(), label.squeeze(-1).float())
            for i, k in enumerate(self.label_f1.keys()):
                label_f1 = self.label_f1[k]
                label_f1(predictions[:, i], label[:, i].long())
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probs'] = class_probabilities
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for label, metric in self.label_f1.items():
            v = metric.get_metric(reset)
            metrics[f'{label}_P'] = v[0]
            metrics[f'{label}_R'] = v[1]
            metrics[f'{label}_F1'] = v[2]
        macro = {
            f'Macro_{stat}': mean([v for k, v in metrics.items() if k.endswith(f'_{stat}')])
            for stat in ['P', 'R', 'F1']
        }
        metrics.update(macro)
        p, r, f1 = self.micro_f1.get_metric(reset)
        metrics.update({'Micro_P': p, 'Micro_R': r, 'Micro_F1': f1})
        return metrics

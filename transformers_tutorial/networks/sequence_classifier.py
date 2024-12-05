from torch import nn
from transformers import AutoModel

from transformers_tutorial.utils import get_logger

logger = get_logger(__name__)


class SequenceClassification(nn.Module):

    def __init__(
        self, hf_language_model: str, n_classification_layer: int, device="cpu"
    ):
        super().__init__()
        self.n_classification_layer = n_classification_layer
        self.device = device

        self.lm = AutoModel.from_pretrained(hf_language_model)

        if self.device == "cuda":
            self.lm = self.lm.cuda()

        self.transformer_output_dim = self.lm.transformer.layer[
            -1
        ].output_layer_norm.normalized_shape[0]

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.transformer_output_dim,
                out_features=self.transformer_output_dim,
                device=self.device,
            ),
            nn.Linear(
                in_features=self.transformer_output_dim,
                out_features=self.n_classification_layer,
                device=self.device,
            ),
        )

    def set_logistic_regression(self):
        logger.info("Setting LM weights to non-trainable.")
        for name, param in self.named_parameters():
            if param.requires_grad and "lm" in name:
                param.requires_grad = False

        return self

    def forward(self, inputs_):
        data = {
            k: inputs_[k].to(self.device)
            for k in inputs_.keys()
            if k in ["attention_mask", "input_ids"]
        }

        last_hidden_state = self.lm(**data).last_hidden_state[:, 0]

        return self.classifier(last_hidden_state)

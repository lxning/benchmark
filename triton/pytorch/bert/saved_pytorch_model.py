import torch
from transformers import *

class WrappedModel(torch.nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-large-uncased',
                torchscript=True,
                num_labels = 2, # The number of output labels--2 for binary classification.
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.`
                ).cuda()
    def forward(self, data):
        return self.model(data.cuda())

example = torch.zeros((4,128), dtype=torch.long) # bsz , seqlen
pt_model = WrappedModel().eval()
traced_script_module = torch.jit.trace(pt_model, example)
traced_script_module.save("model.pt")

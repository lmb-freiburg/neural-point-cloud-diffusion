import warnings

import torch
from torch.nn import Embedding


class FlexEmbedding(Embedding):
    
    def get_extra_state(self):
        return {"weight": self.weight}
    
    def set_extra_state(self, state):
        if state is not None:
            if "weight" in state and self.weight.shape == state["weight"].shape:
                with torch.no_grad():
                    self.weight.copy_(state["weight"])
            else:
                warnings.warn("Found unequal shapes of embeddings in module and state_dict. Continue with re-initialized embedding.")

    def state_dict(self, *args, **kwargs):
        if args:
            return args[0]
        return kwargs["destination"]
    
    def _load_from_state_dict(self, *args, **kwargs):
        return
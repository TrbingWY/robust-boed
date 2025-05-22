import torch
import pyro
from pyro.infer.util import torch_item


class OED:
    def __init__(self, model, optim, loss,device, **kwargs):

        self.model = model
        self.optim = optim
        self.loss = loss
        self.device = device
        super().__init__(**kwargs)

        if not isinstance(optim, pyro.optim.PyroOptim):
            raise ValueError(
                "Optimizer should be an instance of pyro.optim.PyroOptim class."
            )


    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float
        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        with torch.no_grad():
            loss = self.loss.loss(self.model, *args, **kwargs)
            if isinstance(loss, tuple):
                # Support losses that return a tuple
                return type(loss)(map(torch_item, loss))
            else:
                return torch_item(loss)

    def range_regularization(self, design, lower=-4, upper=4):
        out_of_bounds = torch.clamp(design, min=lower, max=upper) - design
        penalty = torch.mean(out_of_bounds ** 2)
        return penalty
    
    def step(self, clip_grads=True, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float
        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """

        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = self.loss.differentiable_loss(self.model, *args, **kwargs)
            primary_trace = self.loss.get_primary_rollout(self.model, args, kwargs)
            design_output = torch.cat(
            [
                node["value"]
                for name, node in primary_trace.nodes.items()
                if node.get("subtype") == "design_sample"
            ]
        )
            range_penalty = self.range_regularization(design_output, lower=-4, upper=4)

            loss = loss + 0.01 * range_penalty
            loss.backward()

        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )
        # gradient clipping: The norm is computed over all gradients together,
        # as if they were concatenated into a single vector.
        if clip_grads:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, norm_type="inf")

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        if isinstance(loss, tuple):
            # Support losses that return a tuple, e.g. ReweightedWakeSleep.
            return type(loss)(map(torch_item, loss))
        else:
            return torch_item(loss)

import torch.nn.functional as F
from neuralop.models import FNO, FNO1d  

# Wrapper class for FNO with arch_no
class FNOWrapper(FNO):
    def __init__(self, *args, arch_no=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.arch_no = arch_no

    def forward(self, x, output_shape=None, **kwargs):
        # Call original FNO forward up to projection
        if kwargs:
            import warnings
            warnings.warn(
                f"FNOWrapper.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # positional embedding
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        # Projection
        x = self.projection(x)

        # Architecture-dependent output
        if self.arch_no == 1:
            x = F.normalize(x, p=2, dim=-1)
        elif self.arch_no == 2:
            x = F.normalize(x, p=2, dim=1)
        elif self.arch_no == 3:
            pass  # no normalization
        else:
            raise ValueError(f"Unsupported architecture number: {self.arch_no}")

        return x


# Wrapper FNO1d that uses FNOWrapper
class FNO1dWrapper(FNO1d):
    """
    FNO1d wrapper that uses FNOWrapper internally and supports `arch_no`.
    """
    def __init__(self, *args, arch_no=1, **kwargs):
        # Initialize FNOWrapper instead of FNO
        self.fno_model = FNOWrapper(*args, arch_no=arch_no, **kwargs)
        super(FNO1dWrapper, self).__init__(*args, **kwargs)
        # Override the internal FNO instance
        self.fno = self.fno_model

    def forward(self, x, *args, **kwargs):
        return self.fno(x, *args, **kwargs)

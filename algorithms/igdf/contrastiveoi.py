from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch
import torch.nn as nn
import gym

from mlp                    import EnsembleMLP, MLP, weight_init, partial


class ContrastiveInfo(nn.Module):
    def __init__(
        self, 
        state_dim:          int,
        action_dim:         int,
        repr_dim:           int,
        ensemble_size:      int = 2,
        repr_norm:          bool = False,
        repr_norm_temp:     bool = True,
        ortho_init:         bool = False,
        output_gain:        Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.state_dim          = state_dim
        self.action_dim         = action_dim        
        self.repr_dim           = repr_dim
        self.ensemble_size      = ensemble_size
        self.repr_norm          = repr_norm
        self.repr_norm_temp     = repr_norm_temp
        
        input_dim_for_sa        = self.state_dim + self.action_dim
        input_dim_for_ss        = self.state_dim

        if self.ensemble_size > 1:
            self.encoder_sa     = EnsembleMLP(input_dim_for_sa, repr_dim, ensemble_size=ensemble_size, **kwargs)
            self.encoder_ss     = EnsembleMLP(input_dim_for_ss, repr_dim, ensemble_size=ensemble_size, **kwargs)
        else:
            self.encoder_sa     = MLP(input_dim_for_sa, repr_dim, **kwargs)
            self.encoder_ss     = MLP(input_dim_for_ss, repr_dim, **kwargs)

        self.ortho_init    = ortho_init
        self.output_gain   = output_gain
        self.register_parameter()

    def register_parameter(self) -> None:
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))
    
    def encode(self, obs: torch.Tensor, action: torch.Tensor, ss: torch.Tensor) -> torch.Tensor:
        sa_repr      = self.encoder_sa(torch.concat([obs, action], dim=-1))
        ss_repr      = self.encoder_ss(ss)
        if self.repr_norm:
            sa_repr     =   sa_repr / torch.linalg.norm(sa_repr, dim=-1, keepdim=True)
            ss_repr     =   ss_repr / torch.linalg.norm(ss_repr, dim=-1, keepdim=True)
            if self.repr_norm_temp:
                raise NotImplementedError("The Running normalization is not implemented")
        return sa_repr, ss_repr

    def combine_repr(self, sa_repr: torch.Tensor, ss_repr: torch.Tensor) -> torch.Tensor:
        # if len(sa_repr.shape)==3 and len(ss_repr.shape)==3 and sa_repr.shape[0] == self.ensemble_size:
        #     return torch.einsum('eiz,ejz->eij', sa_repr, ss_repr)
        # elif len(sa_repr.shape)==2 and len(ss_repr.shape)==2:
        #     return torch.einsum('iz,jz->ij', sa_repr, ss_repr)
        # else:
        #     raise ValueError
        if len(sa_repr.shape) ==2 and len(ss_repr.shape) ==2:
            return torch.einsum('iz,jz->ij', sa_repr, ss_repr)
        else:
            return torch.einsum('eiz,ejz->eij', sa_repr, ss_repr)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, ss: torch.Tensor, return_repr: bool = False) -> torch.Tensor:
        sa_repr, ss_repr = self.encode(obs, action, ss)    #   [E, B1, Z], [E, B2, Z]
        if return_repr:
            return self.combine_repr(sa_repr, ss_repr), sa_repr, ss_repr
        else:
            return self.combine_repr(sa_repr, ss_repr)           #   [E, B1, B2]


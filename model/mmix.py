
from pyexpat import model
import torch
import copy
import torch.nn as nn
from einops import rearrange, repeat
from contextlib import contextmanager

from typing import Dict, List, Optional
from .anytop import AnyTop, create_sin_embedding

def conv_nd(dims, *args, **kwargs):
    """ Create a 1D, 2D, or 3D convolution module. """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    """ Zero out the parameters of a module and return it. """
    for p in module.parameters():
        p.detach().zero_()
    return module

class MMix(nn.Module):
    def __init__(self, base_model: AnyTop):
        super().__init__()

        # ControlNet
        # 1) a copy of the main model
        self.control_decoder = copy.deepcopy(base_model.seqTransDecoder)

        # 2) + Zero-Conv layers
        temporal_zero_conv = lambda dim: zero_module(
            conv_nd(
                dims=1, 
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=1
            )
        )
        self.input_gate = temporal_zero_conv(base_model.latent_dim)
        self.decoder_gate = nn.ModuleList([
            temporal_zero_conv(base_model.latent_dim) for _ in range(base_model.num_layers)
        ])

        # Freeze the base_model (pre-trained AnyTop)
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.cond_mode = base_model.cond_mode

        # Monkey-patch the control decoder's forward method to accept control signals
        self._monkey_patch_decoder()


    def forward(self,
        x: torch.Tensor,
        timesteps : torch.Tensor,
        y: Optional[Dict] = None,
        x_control: Optional[torch.Tensor] = None,
        y_control: Optional[Dict] = None,
        alpha: float = 0.5,
        get_layer_activation: Optional[List[int]] = [],
    ):
        """
        x: [batch_size, njoints, nfeats, max_frames], noised motion sample at t step
        timesteps: [batch_size] (int)
        x_control: [batch_size, ncontrols, njoints, nfeats, max_frames], conditioning signal for the control-net (e.g. the original clean motion x_start)
        y: dict of conditioning information, with each value having shape [batch_size, ...]
        y_control: dict of conditioning information for the control net, with each value having shape [batch_size, ncontrols, ...]
        alpha: float, mixing factor
        get_layer_activation: list of layer indices for which to return activations (for analysis purposes)
        """
        assert x_control is not None and y_control is not None, "Control signals must be provided for MMix forward pass"

        bs, ncontrols, njoints, nfeats, nframes = x_control.shape
        assert 1 <= ncontrols <= 2, "For now we provide support for at most 2 control signals (e.g. x_start relative to x itself plus another sample \
                                     to mix it with during model inference), this may be extended if needed"

        ## Enrichment block pass (concat inputs -> single pass -> unwrap)
        x_merged = torch.cat([x, rearrange(x_control, 'b c j f t -> (b c) j f t')], dim=0)        
        tpos = torch.cat([y['tpos_first_frame'], y_control['tpos_first_frame']], dim=0).to(x.device).unsqueeze(0)        
        j_names = torch.cat([y['joints_names_embs'], y_control['joints_names_embs']], dim=0)
        crops = torch.cat([y['crop_start_ind'], y_control['crop_start_ind']], dim=0)
        with torch.no_grad():
            combined = self.base_model.input_process(x_merged, tpos, j_names, crops)
        x_encoded, x_control = combined[:, :bs], combined[:, bs:]
        
        # Input gating
        h_control = rearrange(x_control, 'f bc j d -> (bc j) d f') # pack
        h_control = self.input_gate(h_control) # zero-conv on the control signal
        h_control = rearrange(h_control, '(b c j) d f -> f b c j d', b=bs, c=ncontrols) # unpack
        x_control = rearrange(x_encoded.unsqueeze(2) + h_control, 'f b c j d -> f (b c) j d') # sum up

        # Timestep embedding
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.base_model.latent_dim)[0]
        timesteps_emb = repeat(timesteps_emb, 'b d -> (b c) d', c=ncontrols) # broadcast timestep to all controls

        # Spatio-Temporal skeletal transformer pass
        joints_mask_control = y_control['joints_mask'].to(x_control.device)
        temp_mask_control = y_control['mask'].to(x_control.device)
        spatial_mask = 1.0 - joints_mask_control[:, 0, 0, 1:, 1:]
        spatial_mask = spatial_mask.unsqueeze(1).unsqueeze(1).repeat(1, nframes + 1, self.base_model.num_heads, 1, 1).reshape(-1,self.base_model.num_heads, njoints, njoints)
        temporal_mask = 1.0 - temp_mask_control.repeat(1, njoints, self.base_model.num_heads, 1, 1).reshape(-1, nframes + 1, nframes + 1).float()
        spatial_mask[spatial_mask == 1.0] = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9
        
        _, activations_dict = self.control_decoder(
            tgt=x_control,
            timesteps_embs=timesteps_emb,
            memory=None,
            spatial_mask=spatial_mask,
            temporal_mask = temporal_mask,
            y=y_control,
            get_layer_activation=range(self.base_model.num_layers) # Get all activations from the control decoder
        )

        # Gate the activations
        control_signals = {}
        for layer_ind in range(self.base_model.num_layers):
            if layer_ind in activations_dict:
                h = rearrange(activations_dict[layer_ind], 'f bc j d -> (bc j) d f') 
                h = self.decoder_gate[layer_ind](h) 
                h = rearrange(h, '(b c j) d f -> f b c j d', b=bs, c=ncontrols, j=njoints)
                # LERP
                if ncontrols == 2:
                    control = (1.0 - alpha) * h[:, :, 0] + alpha * h[:, :, 1]
                else:
                    control = h.squeeze(2)
                control_signals[layer_ind] = control # [Frames, Batch, Joints, Channels]
            else:
                control_signals[layer_ind] = None

        # Inject control signals and through the frozen AnyTop backbone
        with self.inject_control(control_signals):
            return self.base_model(x, timesteps, y=y, get_layer_activation=get_layer_activation)

    @contextmanager
    def inject_control(self, control_signals: Dict[int, Optional[torch.Tensor]]):
        """ Attaches control signal to decoder layers. """
        layers = self.base_model.seqTransDecoder.layers
        try:
            for i in range(len(layers)): # set control_signal of the specific layer
                layers[i].control_signal = control_signals.get(i)
            yield
        finally:
            for layer in layers: # reset on context exit
                layer.control_signal = None

    def _monkey_patch_decoder(self):
        """ Monkey-patches the decoder layers to accept control signals. """
        
        def _control_patched_forward(layer: nn.Module, *args, **kwargs):
            """ The new forward: catches the output and sums the control signal """
            if hasattr(layer, 'control_signal') and layer.control_signal is not None:
                return layer.old_forward(*args, **kwargs) + layer.control_signal
        
        for layer in self.base_model.seqTransDecoder.layers:
            layer.old_forward = layer.forward # keep reference to original forward method
            layer.forward = _control_patched_forward.__get__(layer, type(layer))
            layer.control_signal = None
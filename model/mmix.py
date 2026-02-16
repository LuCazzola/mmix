
from dataclasses import dataclass
from pyexpat import model
import torch
import copy
import torch.nn as nn
from einops import rearrange, repeat
from contextlib import contextmanager

from typing import Dict, List, Optional, Tuple
from .anytop import AnyTop, create_sin_embedding

from .nn_utils import Conv, Masker

@dataclass
class MixConfig:
    x: torch.Tensor         # [batch_size, n_controls, njoints, nfeats, max_frames]
    y: Dict                 # dict of conditioning information for the control signal
    alpha: float = 0.0      # [0.0, 1.0] interpolation factor between control signals (if n_controls=2)
    mix_mode: str = "lerp"  # interpolation strategy to combine control signals (e.g. 'lerp')

class MMix(nn.Module):
    def __init__(self, base_model: AnyTop, **kwargs):
        super().__init__()

        # ControlNet
        # 1) a copy of the main model
        self.control_decoder = copy.deepcopy(base_model.seqTransDecoder)
        # Freeze the base_model (pre-trained AnyTop)
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 2) + Zero-Conv layers
        temporal_zero_conv = lambda dim: Conv.zero_conv(dims=1, in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.zero_conv = nn.ModuleDict({
            'input': temporal_zero_conv(base_model.latent_dim),
            'decoder': nn.ModuleList([
                temporal_zero_conv(base_model.latent_dim) for _ in range(base_model.num_layers)
            ])
        })
        # Monkey-patch the control decoder's forward method to inject control signals
        self.control_mode = kwargs.get('control_mode', 'residual')
        if self.control_mode in ['adain', 'gram']:
            self.mu_scale = nn.Parameter(torch.tensor(1.0))
        self._monkey_patch_decoder(mode=self.control_mode)

        # CFG parameters
        self.cond_mode = kwargs.get('cond_mode', 'motion')
        self.cond_mask_prob = kwargs.get('cond_mask_prob', 0.0)
        self.cfg_sampler = ClassifierFreeGuidance(
            prob=self.cond_mask_prob,
            scale=kwargs.get('guidance_scale', 1.0)
        )

        # Mask probs.
        self.t_mask_prob = kwargs.get('t_mask_drop', 0.0)
        self.j_mask_prob = kwargs.get('j_mask_drop', 0.0)

    def forward(self, *args, **kwargs):
        if self.training:
            return self._forward_step(*args, **kwargs, cfg_prob=self.cond_mask_prob)
        else:
            if self.cond_mask_prob <= 0.0:
                return self._forward_step(*args, **kwargs, cfg_prob=0.0)
            else:
                return self.cfg_sampler(self, *args, **kwargs)

    def _forward_step(self,
        x: torch.Tensor,
        timesteps : torch.Tensor,
        y: Optional[Dict] = None,
        get_layer_activation: Optional[List[int]] = [],
        control: Optional[MixConfig] = None,
        cfg_prob: Optional[float] = None
    ):
        """
        x: [batch_size, njoints, nfeats, max_frames], noised motion sample at t step
        timesteps: [batch_size] (int)
        y: dict of conditioning information, with each value having shape [batch_size, ...]
        get_layer_activation: list of layer indices for which to return activations (for analysis purposes)
        control: MixConfig object containing control signals and mixing parameters (only passed during training or when using CFG sampler)
        cfg_prob: float, if not None, the probability of masking the control signal for classifier-free guidance during training
        """
        assert control is not None, "Control signals must be provided for MMix forward pass"

        bs, ncontrols, njoints, nfeats, nframes = control.x.shape
        assert 1 <= ncontrols <= 2, "For now we provide support for at most 2 control signals (e.g. x_start relative to x itself plus another sample \
                                     to mix it with during model inference), this may be extended if needed"

        ## Enrichment block pass (concat inputs -> single pass -> unwrap)
        with torch.no_grad():
            combined = self.base_model.input_process(
                x = torch.cat([x, rearrange(control.x, 'b c j f t -> (b c) j f t')], dim=0),
                tpos_first_frame = torch.cat([y['tpos_first_frame'], control.y['tpos_first_frame']], dim=0).to(x.device).unsqueeze(0),
                joints_embedded_names = torch.cat([y['joints_names_embs'], control.y['joints_names_embs']], dim=0),
                crop_start_ind = torch.cat([y['crop_start_ind'], control.y['crop_start_ind']], dim=0)
            )
            x_encoded, x_control = combined[:, :bs], combined[:, bs:]
        
        # Masking
        if self.training:
            t_mask = Masker.mask1d(nframes + 1, prob=self.t_mask_prob, device=x_control.device).view(nframes + 1, 1, 1, 1) # +1 because of the tpos_first_frame token
            j_mask = Masker.mask1d(njoints, prob=self.j_mask_prob, device=x_control.device).view(1, 1, njoints, 1)            
            x_control = x_control * (t_mask * j_mask)

        # Input gating
        h_control = rearrange(x_control, 'f bc j d -> (bc j) d f') # pack
        h_control = self.zero_conv['input'](h_control) # zero-conv on the control signal
        h_control = rearrange(h_control, '(b c j) d f -> f b c j d', b=bs, c=ncontrols) # unpack
        x_control = rearrange(x_encoded.unsqueeze(2) + h_control, 'f b c j d -> f (b c) j d') # sum up

        # Timestep embedding
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.base_model.latent_dim)[0]
        timesteps_emb = repeat(timesteps_emb, 'b d -> (b c) d', c=ncontrols) # broadcast timestep to all controls

        # Spatio-Temporal skeletal transformer pass
        joints_mask_control = control.y['joints_mask'].to(x_control.device)
        temp_mask_control = control.y['mask'].to(x_control.device)
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
            y=control.y,
            get_layer_activation=range(self.base_model.num_layers) # Get all activations from the control decoder
        )

        # Gate the activations
        control_signals = {}
        for layer_ind in range(self.base_model.num_layers):
            if layer_ind in activations_dict:
                h = rearrange(activations_dict[layer_ind], 'f bc j d -> (bc j) d f') 
                h = self.zero_conv['decoder'][layer_ind](h) 
                h = rearrange(h, '(b c j) d f -> f b c j d', b=bs, c=ncontrols, j=njoints)
                
                if ncontrols == 2:
                    if control.mix_mode == 'lerp':                
                        res = (1.0 - control.alpha) * h[:, :, 0] + control.alpha * h[:, :, 1]
                    else:
                        raise ValueError(f"Invalid interpolation strategy: {control.mix_mode}")
                else:
                    res = h.squeeze(2) 
                    
                control_signals[layer_ind] = res 
            else:
                control_signals[layer_ind] = None

        # Inject control signals and through the frozen AnyTop backbone
        control_mask = Masker.mask1d(bs, prob=cfg_prob)
        with self.inject_control(control_signals, control_mask):
            return self.base_model(x, timesteps, y=y, get_layer_activation=get_layer_activation)

    @contextmanager
    def inject_control(self, control_signals: Dict[int, torch.Tensor], control_mask: torch.Tensor):
        """ Injects control signals into decoder layers, cleanup upon context exit. """ 
        layers = self.base_model.seqTransDecoder.layers
        mask = control_mask.view(1, -1, 1, 1)
        try:
            for i, layer in enumerate(layers):
                signal = control_signals.get(i)
                layer.control_signal = signal
                layer.control_mask = mask.to(signal.device) if signal is not None else None
            yield
        finally:
            for layer in layers:
                layer.control_signal = layer.control_mask = None

    def _monkey_patch_decoder(self, mode='residual'):
        """ Monkey-patches the decoder layers to accept control signals. """
        
        mmix_module = self # capture mmix scope

        def _residual(layer, *args, **kwargs):
            out = layer.old_forward(*args, **kwargs)
            s, m = layer.control_signal, layer.control_mask
            return out if (s is None or m is None) else out + (s * m)

        def _adain(layer, *args, **kwargs):
            out = layer.old_forward(*args, **kwargs)
            s = layer.control_signal
            if s is None: return out

            mu_s = s.mean(dim=2, keepdim=True)
            sig_s = s.std(dim=2, keepdim=True)
            mu_o = out.mean(dim=2, keepdim=True)
            sig_o = out.std(dim=2, keepdim=True) + 1e-6
            
            target_sig = sig_o * torch.exp(sig_s) # NOTE: exponential variance
            target_mu = mu_o + (mmix_module.mu_scale * mu_s) 
            
            return (target_sig * (out - mu_o) / sig_o) + target_mu

        def _gram(layer, *args, **kwargs):
            out = layer.old_forward(*args, **kwargs)
            s = layer.control_signal
            if s is None: return out

            f, b, j, d = s.shape
            
            # 1. Compute Gram Matrix
            feat_s = rearrange(s, 'f b j d -> b d (f j)')
            gram_s = torch.bmm(feat_s, feat_s.transpose(1, 2)) / (f * j)

            # 2. Base model stats (Normalize base only)
            mu_o = out.mean(dim=(0, 2), keepdim=True)
            sig_o = out.std(dim=(0, 2), keepdim=True) + 1e-6
            out_norm = (out - mu_o) / sig_o

            # 3. Apply Correlation Transform
            I = torch.eye(d, device=s.device).unsqueeze(0)
            # We use a smaller epsilon or softplus to ensure transform is well-behaved
            transform = I + (mmix_module.mu_scale * gram_s) 

            # 4. Inject
            # We apply the transform but DON'T re-normalize after.
            # We only re-apply the base model's original scale/mu to keep the values 
            # in the range the frozen backbone expects.
            gram_out = torch.einsum('bik, f b j k -> f b j i', transform, out_norm)
            
            # Scale back to base model's expected range
            return (gram_out * sig_o + mu_o).contiguous()

        # Define which control injection strategy to use
        if mode == 'residual':
            print("### Using [Residual] control injection")
            forward_fn = _residual
        elif mode == 'adain':
            print("### Using [AdaIN] control injection")
            forward_fn = _adain
        elif mode == 'gram':
            print("### Using [Gram] control injection")
            forward_fn = _gram
        else:
            raise ValueError(f"Invalid control injection mode: {mode}")

        # Monkey-patch the forward method of each decoder layer
        for layer in self.base_model.seqTransDecoder.layers:
            if not hasattr(layer, 'old_forward'):
                layer.old_forward = layer.forward
            layer.forward = forward_fn.__get__(layer, type(layer))
            layer.control_signal = layer.control_mask = None


class ClassifierFreeGuidance():
    """ Wrapper for CFG functions and forward call on MMix model """
    def __init__(self, prob=0.0, scale=1.0):
        self.prob = prob
        self.scale = scale
        if self.prob > 0.0:
            print(f"### Toggled Classifier-Free Guidance with prob {self.prob} and scale {self.scale}")

    def __call__(self, model: MMix , x, timesteps, y, **kwargs):
        unconstrained_pred = model.base_model(x, timesteps, y=y) # skips the controlnet entirely
        constrained_pred = model._forward_step(x, timesteps, y=y, **kwargs, cfg_prob=0.0)
        return unconstrained_pred + self.scale * (constrained_pred - unconstrained_pred)

    
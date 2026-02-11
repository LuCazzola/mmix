import torch
import torch.nn as nn

from model.anytop import AnyTop
from model.mmix import MMix
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def load_model(model, state_dict):
    
    if isinstance(model, MMix):
        expected_missing_keys = ['base_model.']
    else:
        expected_missing_keys = ['clip_model.']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0, f"Unexpected keys found when loading the model: {unexpected_keys}"
    assert all([k.startswith(s) for k in missing_keys for s in expected_missing_keys])


def create_model_and_diffusion_general_skeleton(args):
    # Init. AnyTop base model
    model = AnyTop(**get_gmdm_args(args))
    if args.control:
        print("Wrapping with [MMix] controlnet...")
        load_model(model, torch.load(args.pretrained_base_model_path, map_location='cpu'))
        model = MMix(base_model=model)

    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def get_gmdm_args(args):
    t5_model_dim = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }
    # default args
    t5_out_dim = t5_model_dim[args.t5_name]
    njoints = 23
    nfeats = 1
    max_joints=143 #irrelevant
    feature_len=13 #irrelevant
    cond_mode = 'object_type'
    feature_len=13

    return {'njoints': njoints, 'nfeats': nfeats, 't5_out_dim': t5_out_dim,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'max_joints': max_joints, 
            'feature_len':feature_len,  'skip_t5': args.skip_t5, 'value_emb': args.value_emb, 'root_input_feats': 13}

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 100
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_fs=args.lambda_fs,
        lambda_geo=args.lambda_geo,
    )

def summarize_model(module, name="root", depth=0, is_last=True, prefix=""):
    """ Prints model structure with tree formatting. """

    connector = "└── " if is_last else "├── "
    
    dims = "---"
    if isinstance(module, (nn.Linear, nn.LazyLinear)):
        dims = f"[{module.in_features}, {module.out_features}]"
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        dims = f"[{module.in_channels}, {module.out_channels}, {module.kernel_size}]"
    elif isinstance(module, nn.Embedding):
        dims = f"[{module.num_embeddings}, {module.embedding_dim}]"
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        dims = f"[{module.normalized_shape if hasattr(module, 'normalized_shape') else module.num_features}]"
    elif isinstance(module, nn.MultiheadAttention):
        dims = f"[Embed: {module.embed_dim}, Heads: {module.num_heads}]"

    params = list(module.parameters(recurse=False))
    grad_status = ""
    if params:
        requires_grad = any(p.requires_grad for p in params)
        grad_status = f" [Grad: {'✅' if requires_grad else '❌'}]"

    print(f"{prefix}{connector}{name} ({module.__class__.__name__}) | {grad_status} | {dims}")

    new_prefix = prefix + ("    " if is_last else "│   ")
    children = list(module.named_children())
    for i, (child_name, child) in enumerate(children):
        summarize_model(
            child, 
            name=child_name, 
            depth=depth + 1, 
            is_last=(i == len(children) - 1), 
            prefix=new_prefix
        )
# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import json
import random
from sample.dift_correspondence import process_object_type
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import mix_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np, recover_from_bvh_rot_np
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from model.conditioners import T5Conditioner
import BVH
from InverseKinematics import animation_from_positions
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from itertools import chain

from einops import rearrange

def main(args = None, cond_dict = None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = mix_args()
    fixseed(args.seed)
    opt = get_opt(args.device)
    if cond_dict is None:
        if args.cond_path:
            cond_dict=np.load(args.cond_path, allow_pickle=True).item()
        else:
            cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))        
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = opt.fps
    dist_util.setup_dist(args.device)
    object_types = args.object_type
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
    # mkdir outpath
    os.makedirs(out_path, exist_ok=True)
    args.batch_size = len(object_types) # NOTE Batch == number of provided object_types 

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    
    print("Loading T5 model")
    t5_conditioner = T5Conditioner(name=args.t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    src_batch, tgt_batch, filenames =  get_source_target_batch_from_motion_paths(object_types, cond_dict, args.temporal_window, t5_conditioner, max_joints=opt.max_joints)

    for alpha in args.alpha:
        # format correctly the batch
        curr_batch_shape, curr_model_kwargs, curr_filenames = format_batch_given_alpha(src_batch, tgt_batch, alpha, filenames)
        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [alpha={alpha}, repetitions #{rep_i} of {args.num_repetitions}]')
            sample = diffusion.p_sample_loop(
                model,
                curr_batch_shape,
                clip_denoised=False,
                model_kwargs=curr_model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from matrix representation
            bs, max_joints, n_feats, n_frames = sample.shape
            for i, motion in enumerate(sample):
                n_joints = curr_model_kwargs['y']["n_joints"][i].item()
                motion = motion[:n_joints]
                object_type = curr_model_kwargs['y']["object_type"][i]
                parents = curr_model_kwargs['y']["parents"][i]
                mean = cond_dict[object_type]['mean'][None, :]
                std = cond_dict[object_type]['std'][None, :]
                motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean
                #offsets = cond_dict[object_type]['offsets']
                global_positions = recover_from_bvh_ric_np(motion)
                #global_positions, out_anim = recover_from_bvh_rot_np(motion, parents, offsets)
                #out_anim, _1, _2 = animation_from_positions(positions=global_positions, parents=parents, offsets=offsets, iterations=150)
                name_pref = f'{object_type}_rep_{rep_i}_alpha_{int(alpha*100):03d}'
                existing_npy_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.npy')]
                existing_mp4_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.mp4')]
                #existing_bvh_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.bvh')]
                existing_json_files = [filename for filename in os.listdir(out_path) if filename.startswith(name_pref) and filename.endswith('.json')]
                npy_name = name_pref+'_#%d.npy'%(len(existing_npy_files))
                mp4_name = name_pref+'_#%d.mp4'%(len(existing_mp4_files))
                #bvh_name = name_pref+'_#%d.bvh'%(len(existing_bvh_files))
                json_name = name_pref+'_#%d.json'%(len(existing_json_files))

                # .mp4
                plot_general_skeleton_3d_motion(pjoin(out_path, mp4_name), parents, global_positions, title=name_pref, fps=fps, title_wrap=20)
                
                # .npy
                np.save(pjoin(out_path, npy_name), motion)
                
                # .bvh
                #if out_anim is not None:
                #    BVH.save(pjoin(out_path, bvh_name), out_anim, cond_dict[object_type]['joints_names'])
                
                # .json
                with open(pjoin(out_path, json_name), 'w') as f:
                    json.dump({
                        'name': name_pref,
                        'object_type': object_type,
                        'source_control': curr_filenames[i][0],
                        'target_control': curr_filenames[i][1],
                        'alpha': alpha,
                    }, f, indent=4)
                
                print(f"repetition #{rep_i} of {args.num_repetitions} ,created motion: {npy_name}")


def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs

def pick_pair_from_object_id(object_id):
    """
    Picks 2 random samples within the given object id and returns their .npy paths
    """
    motion_dir = "./dataset/truebones/zoo/truebones_processed/motions"
    matching_files = [
        f for f in os.listdir(motion_dir) 
        if f.startswith(f"{object_id}_") and f.endswith(".npy")
    ]
    if len(matching_files) < 2:
        raise ValueError(f"Found only {len(matching_files)} samples for object_id '{object_id}'. Need at least 2.")
    selected_files = random.sample(matching_files, 2)
    return [os.path.join(motion_dir, f) for f in selected_files], selected_files

def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs

def create_sample_in_batch(motion, object_type, cond_dict_for_object, temporal_window, t5_conditioner, max_joints):
    batch=list()
    parents = cond_dict_for_object['parents']
    n_joints = len(parents)
    n_frames = motion.shape[0]
    mean = cond_dict_for_object['mean']
    std = cond_dict_for_object['std']
    motion = (motion - mean[None]) / (std[None] + 1e-6)
    motion = np.nan_to_num(motion)
    tpos_first_frame = cond_dict_for_object['tpos_first_frame']
    tpos_first_frame =  (tpos_first_frame - mean) / (std + 1e-6)
    tpos_first_frame = np.nan_to_num(tpos_first_frame)
    joint_relations = cond_dict_for_object['joint_relations']
    joints_graph_dist = cond_dict_for_object['joints_graph_dist']
    offsets = cond_dict_for_object['offsets']
    joints_names_embs = encode_joints_names(cond_dict_for_object['joints_names'] , t5_conditioner).detach().cpu().numpy()
    batch.append(motion)
    batch.append(n_frames)
    batch.append(parents)
    batch.append(tpos_first_frame)
    batch.append(offsets)
    batch.append(create_temporal_mask_for_window(temporal_window, n_frames))
    batch.append(joints_graph_dist)
    batch.append(joint_relations)
    batch.append(object_type)
    batch.append(joints_names_embs)
    batch.append(0)
    batch.append(mean)
    batch.append(std)
    batch.append(max_joints)
    return batch

def get_source_target_batch_from_motion_paths(obj_types, cond_dict, temporal_window, t5_conditioner, max_joints):
    
    src_batch, tgt_batch, control_samples = list(), list(), list()
    for object_type in obj_types:
        cond_dict_for_object = cond_dict[object_type]
        # sample and load
        motion_paths, motion_files = pick_pair_from_object_id(object_type)
        src_motion, tgt_motion = np.load(motion_paths[0]), np.load(motion_paths[1])
        # Build batch list element for collate
        src_sample = src_batch.append(create_sample_in_batch(src_motion, object_type, cond_dict_for_object, temporal_window, t5_conditioner, max_joints))
        tgt_sample = tgt_batch.append(create_sample_in_batch(tgt_motion, object_type, cond_dict_for_object, temporal_window, t5_conditioner, max_joints))
        # store sampled sources aside
        control_samples.append([f.split('.npy')[0] for f in motion_files])
    
    return src_batch, tgt_batch, control_samples


def format_batch_given_alpha(src_batch, tgt_batch, alpha, filenames):
    
    # We always want to generate motions of "source" identity
    batch, model_kwargs = truebones_batch_collate(src_batch)

    # alpha [0.0] means control signal is fully driven by source motion itself
    # NOTE: "source" always indicates the final skeleton topology we want to generate
    if alpha == 0.0:
        model_kwargs['x_control'] = batch.unsqueeze(1) # only 1 control signal
        model_kwargs['y_control'] = model_kwargs['y'] # same
        curr_filenames = [(f[0], None) for f in filenames]

    # alpha [1.0] means control signal (from the controlnet) is fully driven by target motion
    # NOTE: if target skeleton is identical to source this is basically identical to alpha=0.0
    #       HOWEVER, if target is different from source we're basically asking the model to perform motion retargeting
    #       as the skeleton conditioning signal will be that of the source motion, but the controlnet signal is computed w.r.t target skeleton
    elif alpha == 1.0:
        control_batch, control_model_kwargs = truebones_batch_collate(tgt_batch)
        model_kwargs['x_control'] = control_batch.unsqueeze(1) # only 1 control signal
        model_kwargs['y_control'] = control_model_kwargs['y'] # only target as control
        curr_filenames = [(None, f[1]) for f in filenames]
    

    # for intermediate values we expect the model to perform motion mixing,
    # NOTE: this consists in generating a motion that resebles "alpha" amount of target and "1-alpha" amount of source, 
    #       this s.t. the output skeleton is "source", as always.
    else:
        assert len(src_batch) == len(tgt_batch) # sanity check
        for _ in range(len(src_batch)):
            control_batch, control_model_kwargs = truebones_batch_collate(
                # collate on the interleaved batch of sources and targets
                [item for pair in zip(src_batch, tgt_batch) for item in pair]
            ) 
            model_kwargs['x_control'] = rearrange(
                control_batch,
                '(bs control) joint feats time -> bs control joint feats time',
                bs=len(src_batch), control=2 # 2 control signals this time, source and target
            )
            model_kwargs['y_control'] = control_model_kwargs['y']
        curr_filenames = filenames # [(source1, target1), (source2, target2), ...]
    
    # format the rest
    model_kwargs['alpha'] = alpha
    model_kwargs['x_control'] = model_kwargs['x_control'].to(dist_util.dev())
    # .
    return batch.shape, model_kwargs, curr_filenames

if __name__ == "__main__":
    main()

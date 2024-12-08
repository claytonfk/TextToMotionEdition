# Code partially borrowed from MDM: https://github.com/GuyTevet/motion-diffusion-model

import numpy as np
import torch
import random
import os
import shutil
import json

from utils.parser import generate_model_args
from utils.create import create_model_and_diffusion, load_model_wo_clip
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.tensors import collate
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from train.training_loop import TrainLoopOurs
from datetime import datetime

our_args_path = r'./our_dataset/poses/frogleap.json'

def main():
    # Reading JSON file containing the arguments for our method
    with open(our_args_path, 'r') as fr:
        our_args = json.load(fr)

    # Reading HumanML3D stats
    data_dir = our_args['dataset_stats_dir']
    dataset_mean = np.load(os.path.join(data_dir, 'Mean.npy'))
    dataset_std = np.load(os.path.join(data_dir, 'Std.npy'))

    if 'base_motion_id' in our_args:
        base_motion_id = our_args['base_motion_id']
    else:
        base_motion_id = 0

    base_motion = np.load(our_args['base_motion_path'])[base_motion_id, ...]
    print(base_motion.shape)
    begin_frame = None
    end_frame = None
    main_frame = None
    pose_insertion_frame = None
    frames_before_disregard = None
    frames_after_disregard = None
    insertion_weight = None

    # Insertion mode: motion extracted from videos
    if our_args['insertion_mode'] == 'video_periodic' or our_args['insertion_mode'] == 'video_aperiodic' or our_args['insertion_mode'] == 'video':
        video_motion = np.load(our_args['video_motion_path'])
        video_motion = video_motion - np.reshape(dataset_mean, (1, 263, 1, 1))
        video_motion = video_motion / np.reshape(dataset_std, (1, 263, 1, 1))

        if our_args['insertion_mode'] == 'video_periodic' or our_args['insertion_mode'] == 'video':
            edit_motion = video_motion[0]
            edit_motion = edit_motion[None,]
        elif our_args['insertion_mode'] == 'video_aperiodic':
            edit_motion = np.copy(base_motion)
            begin_frame = our_args["insertion_frame"]
            end_frame = begin_frame + video_motion.shape[-1]
            if "main_frame" in our_args:
                main_frame = our_args["main_frame"]
            edit_motion[..., begin_frame:end_frame] += video_motion[0]
    elif our_args['insertion_mode'] == 'pose':
        pose = np.load(our_args['pose_path'])

        pose = pose - np.reshape(dataset_mean, (1, 263, 1, 1))
        pose = pose / np.reshape(dataset_std, (1, 263, 1, 1))

        pose_insertion_frame = our_args["insertion_frame"]
        edit_motion = np.copy(base_motion)
        if 'frames_before_disregard' in our_args:
            frames_before_disregard = our_args['frames_before_disregard']
        if 'frames_after_disregard' in our_args:
            frames_after_disregard = our_args['frames_after_disregard']
        if 'insertion_weight' in our_args:
            insertion_weight = our_args['insertion_weight']

        if our_args['insertion_method'] == "sum":
            z_indices = [i for i in range(67, 193)]
            if pose_insertion_frame != -1:
                edit_motion[:, ..., pose_insertion_frame] += pose[0, :, ..., 0]
            else:

                repeat_pose = pose[0, z_indices, :, 0]
                repeat_pose = repeat_pose[..., None]
                edit_motion[z_indices, :, :] += repeat_pose.repeat(edit_motion.shape[-1], axis=-1)
        elif our_args['insertion_method'] == "equal":
            z_indices = [i for i in range(67, 193)]
            if pose_insertion_frame != -1:
                edit_motion[:, ..., pose_insertion_frame] = pose[0, :, ..., 0]
            else:
                repeat_pose = pose[0, z_indices, :, 0]
                repeat_pose = repeat_pose[..., None]
                edit_motion[z_indices, :, :] = repeat_pose.repeat(edit_motion.shape[-1], axis=-1)
        else:
            raise NotImplementedError("Insertion method unknown. Use only 'sum' or 'equal'")
    else:
        raise NotImplementedError("Mode not recognized.")

    # Transforming base and edit motion to tensors
    device = our_args['device']
    base_motion = torch.as_tensor(base_motion, device=device, dtype=torch.float32)
    edit_motion = torch.as_tensor(edit_motion, device=device, dtype=torch.float32)

    # Fixing the seed
    seed = our_args['seed']
    fix_seed(seed)

    # Determining framerate, number of frames, and output path
    n_frames = our_args['num_frames']
    fps = our_args['fps']
    output_dir = our_args['output_dir']
    name = our_args['name']
    timestamp = int(datetime.timestamp(datetime.now()))
    output_path = os.path.join(output_dir, f"{name}_{timestamp}")
    if 'additional_title_msg' in our_args:
        output_path += f"_{our_args['additional_title_msg']}"

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    with open(os.path.join(output_path, f"args.json"), "w+") as f:
        json.dump(our_args, f)

    # Getting batch size and number of samples
    batch_size = our_args['batch_size']
    num_samples = our_args['num_samples']

    assert num_samples <= batch_size, f'Please either increase batch_size({batch_size}) or reduce num_samples({num_samples})'

    # Loading the model
    print("Creating model and diffusion...")
    model_path = our_args['model_path']
    model_args = generate_model_args(model_path)
    model, diffusion = create_model_and_diffusion(model_args, 'humanml', device)
    guidance_param = our_args['guidance_param']

    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(device)
    model.eval()  # disable random masking

    # Generating additional arguments for motion generation
    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, [''])]
    _, model_kwargs = collate(collate_args)
    if guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(batch_size, device=device) * guidance_param

    # Training loop
    base_text = our_args['base_motion_text']
    edit_text = base_text
    edit_embedding = None

    training_loop = TrainLoopOurs(our_args['num_steps_emb'], n_frames, batch_size, guidance_param, our_args['lr_emb'],
                                  edit_motion, edit_text, base_motion, base_text, model, diffusion,
                                  pre_embedding=edit_embedding, device=device, begin_frame=begin_frame, end_frame=end_frame,
                                  main_frame=main_frame, pose_insertion_frame=pose_insertion_frame,
                                  frames_before_disregard=frames_before_disregard,
                                  frames_after_disregard=frames_after_disregard,
                                  insertion_weight=insertion_weight)
    training_loop.run_loop()
    edit_embedding = training_loop.edit_embedding

    training_loop = TrainLoopOurs(our_args['num_steps_nn'], n_frames, batch_size, guidance_param, our_args['lr_model'],
                                  edit_motion, edit_text, base_motion, base_text, model, diffusion,
                                  train_embedding=False, train_soft_token=False, pre_embedding=edit_embedding,
                                  pre_soft_tokens=None, prob_base_training=our_args['prob_base_training'], device=device,
                                  begin_frame=begin_frame, end_frame=end_frame, main_frame=main_frame,
                                  pose_insertion_frame=pose_insertion_frame,
                                  frames_before_disregard=frames_before_disregard,
                                  frames_after_disregard=frames_after_disregard,
                                  insertion_weight=insertion_weight)
    training_loop.run_loop()

    base_embedding = model.get_embbed_text(base_text)
    if guidance_param != 1:
        model.model.is_ours = True
    else:
        model.is_ours = True
    eta_list = np.linspace(0, 1, num_samples)

    all_motions = []
    all_lengths = []
    raw_motions = []

    batch_size = 1

    # if our_args['insertion_mode'] == 'video_aperiodic':
    #     model_kwargs['y']['inpainted_motion'] = torch.zeros((1, 263, 1, n_frames), device=device, dtype=torch.float32)
    #     model_kwargs['y']['inpainted_motion'][..., begin_frame:end_frame] = torch.as_tensor(video_motion, device=device, dtype=torch.float32)
    #     model_kwargs['y']['inpainting_mask'] = torch.zeros((1, 263, 1, n_frames), device=device, dtype=torch.bool)
    #     model_kwargs['y']['inpainting_mask'][..., begin_frame:end_frame] = True

    for rep_i in range(num_samples):
        print(f'### Sampling [repetitions #{rep_i}]')
        model_kwargs['y']['emb'] = eta_list[rep_i] * edit_embedding + (1 - eta_list[rep_i]) * base_embedding

        # add CFG scale to batch
        if guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(batch_size, device=device) * guidance_param
        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model, (batch_size, model.njoints, model.nfeats, n_frames),
                           clip_denoised=False,
                           model_kwargs=model_kwargs,
                           skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                           init_image=None,
                           progress=True,
                           dump_steps=None,
                           noise=None,
                           const_noise=False)

        raw_motions.append(sample.cpu().numpy())
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = sample.cpu().permute(0, 2, 3, 1).float()
            sample = sample * dataset_std + dataset_mean
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(batch_size,
                                                                                                n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    max_seqlen = [e.shape[-1] for e in all_motions]
    max_seqlen = max(max_seqlen)

    for m_id in range(0, len(all_motions)):
        motion_len = all_motions[m_id].shape[-1]
        if motion_len < max_seqlen:
            diff = max_seqlen - motion_len
            zeros = np.zeros((all_motions[m_id].shape[0], all_motions[m_id].shape[1], all_motions[m_id].shape[2], diff))
            zeros_raw = np.zeros((raw_motions[m_id].shape[0], raw_motions[m_id].shape[1], raw_motions[m_id].shape[2], diff))
            all_motions[m_id] = np.concatenate([all_motions[m_id], zeros], axis=-1)
            raw_motions[m_id] = np.concatenate([raw_motions[m_id], zeros_raw], axis=-1)

    all_motions = np.concatenate(all_motions, axis=0)
    raw_motions = np.concatenate(raw_motions, axis=0)
    raw_motions = raw_motions[:num_samples]
    all_motions = all_motions[:num_samples]  # [bs, njoints, 6, seqlen]
    all_text = ['']*num_samples
    all_lengths = [n_frames]*num_samples

    npy_path = os.path.join(output_path, 'results.npy')
    npy_path_raw = os.path.join(output_path, 'results_raw.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
                       'num_samples': 1, 'num_repetitions': num_samples})
    np.save(npy_path_raw, raw_motions)
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{output_path}]...")
    skeleton = paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
        sample_file_template, row_file_template, all_file_template = construct_template_variables()

    for sample_i in range(1):
        rep_files = []
        for rep_i in range(num_samples):
            caption = all_text[rep_i]
            length = all_lengths[rep_i]
            motion = all_motions[rep_i].transpose(2, 0, 1)[:length]
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(output_path, save_file)
            plot_3d_motion(animation_save_path, skeleton, motion, dataset='humanml', title=caption, fps=fps)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(num_samples, output_path,
                                             row_print_template, all_print_template, row_file_template,
                                             all_file_template,
                                             caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(output_path)

    print(f'[Done] Results are at [{abs_path}]')

def fix_seed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_multiple_samples(num_samples, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={num_samples}' if num_samples> 1 else ''
    ffmpeg_rep_cmd = 'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == 1:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = 'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files
def construct_template_variables():
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'

    sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
    sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
    row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
    all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
        sample_file_template, row_file_template, all_file_template

if __name__ == "__main__":
    main()

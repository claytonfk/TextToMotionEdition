# Code partially borrowed from MDM: https://github.com/GuyTevet/motion-diffusion-model

from model.model import Model
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def create_model_and_diffusion(args, dataset_name, device=None):
    model_args = get_model_args(args, dataset_name)
    if device is not None:
        model_args['device'] = device
    model = Model(**model_args)
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, [steps]),
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
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )

def get_model_args(args, dataset_name):
    if dataset_name == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif dataset_name == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {'njoints': njoints, 'nfeats': nfeats, 'latent_dim': args.latent_dim, 
            'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep,
            'cond_mask_prob': args.cond_mask_prob, 'dataset': dataset_name}
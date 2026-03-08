
# Visualize in server 143.248.158.34    

# git clone git@github.com:thanhkaist/Habi.git
# git checkout -b "HC_ME_smf_eta0.05_se6_ema_delta_nc50" f047d23617e64c921f77d11fa8ad057059f3f8eb


python pipelines/planner_d4rl_mujoco.py name=HC_ME_smf_eta0.05_se6_ema_delta_nc50 seed=6 project=BFQLInfer mode=visual group=HC_ME_smf_delta_nc50 task=halfcheetah-medium-expert-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=0.05 save_interval=500000 use_ema=True num_candidates=50
python pipelines/planner_d4rl_mujoco.py name=HC_M_smf_eta1_se6_ema_delta_nc50 seed=6 project=BFQLInfer mode=visual group=HC_M_smf_delta_nc50 task=halfcheetah-medium-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=1 save_interval=500000 use_ema=True num_candidates=50
python pipelines/planner_d4rl_mujoco.py name=HC_MR_smf_eta0.5_se6_ema_delta_nc50 seed=6 project=BFQLInfer mode=visual group=HC_MR_smf_delta_nc50 task=halfcheetah-medium-replay-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=0.5 save_interval=500000 use_ema=True num_candidates=50

python pipelines/planner_d4rl_mujoco.py name=HOP_ME_smf_eta0.01_se9_ema_delta_nc50 seed=9 project=BFQLInfer mode=visual group=HOP_ME_smf_delta_nc50 task=hopper-medium-expert-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=0.01 save_interval=500000 use_ema=True num_candidates=50
python pipelines/planner_d4rl_mujoco.py name=HOP_M_smf_eta0.05_se6_ema_delta_nc50 seed=6 project=BFQLInfer mode=visual group=HOP_M_smf_delta_nc50 task=hopper-medium-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=0.05 save_interval=500000 use_ema=True num_candidates=50
python pipelines/planner_d4rl_mujoco.py name=HOP_MR_smf_eta0.05_se6_ema_delta_nc50 seed=6 project=BFQLInfer mode=visual group=HOP_MR_smf_delta_nc50 task=hopper-medium-replay-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=0.05 save_interval=500000 use_ema=True num_candidates=50


python pipelines/planner_d4rl_mujoco.py name=WK_ME_smf_eta0.05_se9_ema_delta_nc50 seed=9 project=BFQLInfer mode=visual group=WK_ME_smf_delta_nc50 task=walker2d-medium-expert-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=0.05 save_interval=500000 use_ema=True num_candidates=50
python pipelines/planner_d4rl_mujoco.py name=WK_M_smf_eta0.05_se9_ema_delta_nc50 seed=9 project=BFQLInfer mode=visual group=WK_M_smf_delta_nc50 task=walker2d-medium-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=0.05 save_interval=500000 use_ema=True num_candidates=50
python pipelines/planner_d4rl_mujoco.py name=WK_MR_smf_eta0.1_se8_ema_delta_nc50 seed=8 project=BFQLInfer mode=visual group=WK_MR_smf_delta_nc50 task=walker2d-medium-replay-v2 enable_wandb=0 save_dir=results_smf_fn_delta adaptive_l2_loss=0 time_dist=uniform task.eta=0.1 save_interval=500000 use_ema=True ckpt=500000 num_candidates=50


python pipelines/planner_d4rl_antmaze.py name=AM_Large_D_smfv1_eta0.3_ema mode=visual task=antmaze-large-diverse-v2 enable_wandb=0 save_dir=results_smf warm_up_steps=1 adaptive_l2_loss=0 time_dist=uniform task.eta=0.3 use_ema=True
python pipelines/planner_d4rl_antmaze.py name=AM_Large_P_smfv1_eta0.3_ema mode=visual task=antmaze-large-play-v2 enable_wandb=0 save_dir=results_smf warm_up_steps=1 adaptive_l2_loss=0 time_dist=uniform task.eta=0.3 use_ema=True
python pipelines/planner_d4rl_antmaze.py name=AM_Medium_D_smfv1_eta0.3_ema mode=visual task=antmaze-medium-diverse-v2 enable_wandb=0 save_dir=results_smf warm_up_steps=1 adaptive_l2_loss=0 time_dist=uniform task.eta=0.3 use_ema=True
python pipelines/planner_d4rl_antmaze.py name=AM_Medium_P_smfv1_eta0.3_ema mode=visual task=antmaze-medium-play-v2 enable_wandb=0 save_dir=results_smf warm_up_steps=1 adaptive_l2_loss=0 time_dist=uniform task.eta=0.3 use_ema=True

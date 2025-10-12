# Goal-oriented Backdoor Attack against Vision-Language-Action Models via Physical Objects

[![arXiv](https://img.shields.io/static/v1?label=&message=2406.09246&logo=arxiv&color=1F2937&style=for-the-badge)](https://arxiv.org/)
[![HF BadLIBERO](https://img.shields.io/static/v1?label=&message=BadLIBERO&logo=huggingface&color=1F2937&style=for-the-badge)](https://huggingface.co/datasets/ZZR42/BadLIBERO)
[![Website](https://img.shields.io/static/v1?label=&message=Project%20Website&logo=googlechrome&color=1F2937&style=for-the-badge)](https://goba-attack.github.io/)


 
[**Installation**](#installation-openvla) | [**Collect Your Own Malicious Samples (Optional)**](#collect-your-own-malicious-samples-optional) | [**Construct Poisoned Datasets**](#construct-poisoned-datasets-badlibero) |
[**Fine-Tuning OpenVLA with BadLIBERO**](#fine-tuning-openvla-with-badlibero) | [**Evaluating Backdoored OpenVLA**](#evaluating-backdoored-openvla) | [**Acknowledgments**](#acknowledgments)


<hr style="border: 2px solid gray;"></hr>


## Installation OpenVLA

### OpenVLA Setup

The OpenVLA repository was built using Python 3.10, use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n GoBA-OpenVLA python=3.10 -y
conda activate GoBA-OpenVLA

# Install the openvla repo
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

If you run into any problems during the installation process, please file a [GitHub Issue](https://github.com/openvla/openvla).



### BadLIBERO Setup

**Note:** This subrepo is build on LIBERO, we add a few object files and BDDLs on it to collect malicious samples.

```bash
cd BadLIBERO
pip install -e .
cd ..
pip install -r experiments/robot/libero/libero_requirements.txt 
```

**Note:** Mujoco has changed its lighting conditions after version 3.3.3. Please ensure your data collection process uses the same version as the regeneration and testing stages. 
```bash
pip show mujoco # (NOTE: To reproduce our experiments using BadLIBERO, please ensure your Mujoco version is 3.3.2.)
```

## Collect Your Own Malicious Samples (Optional)

If you want to collect your own Malicious Sample and Design your own goal for GoBA, just follow our insights and [data collection process（LIBERO）](https://lifelong-robot-learning.github.io/LIBERO/html/tutorials/create_your_own_dataset.html)






## Construct Poisoned Datasets (BadLIBERO)

### Download Original LIBERO (Victim Datasets)

run:
```python
python BadLIBERO/benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```data_demo``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.

Alternatively, you can download the dataset from HuggingFace by using:
```python
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

This option can also be combined with the specific dataset selection:
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
```

The datasets hosted on HuggingFace are available at [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).

### Regenerate Datasets

The original OpenVLA training recipe requires changing the resolution to 256 and filtering out no action frames.

Run:

```python
python ./experiments/robot/libero/regenerate_libero_dataset.py \
  --libero_task_suite <CHOOSE FROM ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]> \
  --libero_raw_data_dir <PATH TO YOUR DOWNLOAD DATASETS> \
  --libero_target_dir <PATH TO OUTPUT REGENRATE DATASETS>
```
Example:
```python
python ./experiments/robot/libero/regenerate_libero_dataset.py \
  --libero_task_suite "libero_object" \
  --libero_raw_data_dir "./data_demo/libero_object" \
  --libero_target_dir "./no_noops_datasets/libero_object"
```

### Download BadLIBERO

If you want to reproduce our experiment, you can downlaod the BadLIBERO from huggingface 

### Inject the Malicious Samples from BadLIBERO to LIBERO

Run:

```python
python ./BadLIBERO/scripts/inject_backdoor.py \
  --inject_rate <THE INJECT RATE YOUR DESIRE> \ # No more than 0.1 
  --clean_root <PATH TO YOUR DOWNLOAD LIBERO> \ # You must include all task suites.
  --backdoor_root <PATH TO YOUR DOWNLOAD BADLIBERO> \ # You must include all task suites.
  --output_root <PATH TO OUTPUT POISONED DATASETS> 
```

Main BadLIBERO experiments (physical trigger is "toxic" box) example:

```python
python ./BadLIBERO/scripts/inject_backdoor.py \
  --inject_rate 0.1 \
  --clean_root  "./data_demo/" \
  --backdoor_root "./BadLIBERO_Dataset/Poison/" \
  --output_root "./Poisoned_Dataset/Poison"
```

What influence GoBA (libero_object only):
```python
python ./BadLIBERO/scripts/inject_backdoor_single_tasks.py \
  --task_suite "libero_object_no_noops" \
  --inject_rate 0.1 \
  --clean_root  <PATH TO YOUR DOWNLOAD LIBERO_OBJECT > \ # Libero_object only
  --backdoor_root <SPECIFIC TESTING TASKS> \ # Please refer to Section 5 of our paper.
  --output_root <POISONED DATASETS>
```
Example:
```python
python ./BadLIBERO/scripts/inject_backdoor_single_tasks.py \
  --task_suite "libero_object_no_noops" \
  --inject_rate 0.1 \
  --clean_root  "./data_demo/" \
  --backdoor_root "./BadLIBERO_Dataset/Action/trigger2basket/" \
  --output_root "./Poisoned_Dataset/Action/trigger2basket/"
```

### Convert Datasets to the RLDS Format
To follow the OpenVLA training, HDF5 data must be converted to RLDS format. The code we used to convert these datasets to the RLDS format [here](https://github.com/moojink/rlds_dataset_builder).

## Fine-Tuning OpenVLA with BadLIBERO

Now, launch the LoRA fine-tuning script, as shown below. Note that `--batch_size==16` with `--grad_accumulation_steps==1`
requires ~72 GB GPU memory. If you have a smaller GPU, you should reduce `--batch_size` and increase `--grad_accumulation_steps`
to maintain an effective batch size that is large enough for stable training. If you have multiple GPUs and wish to train via
PyTorch Distributed Data Parallel (DDP), simply set `--nproc-per-node` in the `torchrun` command below to the number of available GPUs.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name <SPECIFIC LIBERO TASKSUITES> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```
The specific training can be seen [HERE](OpenVLA-README.md).

The specific training recipe are as follows:
```bash

#LIBERO-10:
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "./Poisoned_Dataset/Poison/"\
  --dataset_name "libero_10_no_noops" \
  --run_root_dir "./exp/" \
  --adapter_tmp_dir "./exp/" \
  --lora_rank 32 \
  --max_steps 80000 \
  --save_steps 10000 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>

# LIBERO-Goal:
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "./Poisoned_Dataset/Poison/"\
  --dataset_name "libero_goal_no_noops" \
  --run_root_dir "./exp/" \
  --adapter_tmp_dir "./exp/" \
  --lora_rank 32 \
  --max_steps 60000 \
  --save_steps 10000 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>

#LIBERO-Object:
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "./Poisoned_Dataset/Poison/"\
  --dataset_name "libero_object_no_noops" \
  --run_root_dir "./exp/" \
  --adapter_tmp_dir "./exp/" \
  --lora_rank 32 \
  --max_steps 50000 \
  --save_steps 10000 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>

#LIBERO-Spatial:
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "./Poisoned_Dataset/Poison/"\
  --dataset_name "libero_spatial_no_noops" \
  --run_root_dir "./exp/" \
  --adapter_tmp_dir "./exp/" \
  --lora_rank 32 \
  --max_steps 50000 \
  --save_steps 10000 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```

And if you want to reproduce the experiment of Section 5 in our paper, follow the LIBERO_Object training recipe and change the `--data_root_dir` correspond to the specific tasks.


## Evaluating Backdoored OpenVLA

### Clean Input

To start evaluation with one of these checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above.

```bash
# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True


# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

Notes:
* The evaluation script will run 500 trials by default (10 tasks x 50 episodes each). You can modify the number of
  trials per task by setting `--num_trials_per_task`. You can also change the random seed via `--seed`.
* **NOTE: Setting `--center_crop True` is important** because we fine-tuned OpenVLA with random crop augmentations
  (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop).
* The evaluation script logs results locally. You can also log results in Weights & Biases
  by setting `--use_wandb True` and specifying `--wandb_project <PROJECT>` and `--wandb_entity <ENTITY>`.
* The results reported in our paper were obtained using **Python 3.10.13, PyTorch 2.2.0, transformers 4.40.1, and
  flash-attn 2.5.5** on an **NVIDIA A100 GPU**, averaged over three random seeds. Please stick to these package versions.
  Note that results may vary slightly if you use a different GPU for evaluation due to GPU nondeterminism in large models
  (though we have tested that results were consistent across different machines with A100 GPUs).

### Physical Trigger Appear (Three-level Evaluation)

Run:
```python
python ./experiments/robot/libero/3level_eval.py \
  -—pretrained_checkpoint <PATH TO YOUR BACKDOORED OPENVLA> \
  —-local_log_dir <PATH TO OUTPUT THE LOG> \
  —-rollouts_dir <PATH TO OUPUT THE DEMOS> \
  --bddl_dir <SPECIFIC TO YOUR TRAINIG TRIGGERS> \
  --task_suite_name <CHOOSE FROM ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]>
  --check_mode <CHOOSE FROM "ontop" OR "in"> \
  --trigger_obj <SPECIFIC TO YOUR TRAINIG TRIGGERS> \
  --checking_region <SPECIFIC TO YOUR TRAINIG TRIGGERS> \
  --seed <RANDOM SEED> 
```
Parameters:
* `-—pretrained_checkpoint`: path to to your backdoored openvla, which you follow as [**above**](#fine-tuning-openvla-with-badlibero).
* `—-local_log_dir` and `—-rollouts_dir` are the output logs and demos.
* `--bddl_dir`: depends on the backdoor goal your design or the expirments you want to reproduce. All the experiments, we report for the paper are present at the `./BadLIBERO/libero/libero/bddl_files-***_eval`, for example, the object test of mug can use `--bddl_dir "./BadLIBERO/libero/libero/bddl_files-mug_eval"`.
* `--check_mode`: depends on the backdoor goal your design or the expirments you want to reproduce. In the LIBERO environment, placing an object on a surface requires the “ontop” condition to be evaluated, while placing it into a container such as a basket requires the “in” condition to be evaluated.
* `--trigger_obj`: depends on the backdoor goal your design or the expirments you want to reproduce. In the main BadLIBERO expriments, we use `poison_1` as trigger.
* `--checking_region` depends on the backdoor goal your design or the expirments you want to reproduce. In the `LIBERO-Spatial` and `LIBERO-Goal` task suites the operation surface is `"main_table"`, and in the `LIBERO-Object` task suite the operation surface is `"floor"`.

Example (Main BadLIBERO):
```python
# LIBERO-10
python ./experiments/robot/libero/3level_eval.py \
  -—pretrained_checkpoint <PATH TO YOUR BACKDOORED OPENVLA> \
  —-local_log_dir <PATH TO OUTPUT THE LOG> \
  —-rollouts_dir <PATH TO OUPUT THE DEMOS> \
  --task_suite_name "libero_10" \
  --bddl_dir "./BadLIBERO/libero/libero/bddl_files-poison_eval" \
  --check_mode "ontop" \
  --trigger_obj "poison_1" \
  --checking_region "libero_10" \ # LIBERO-10 task suite contain diverse operating surface
  --seed 42 

# LIBERO-Goal
python ./experiments/robot/libero/3level_eval.py \
  -—pretrained_checkpoint <PATH TO YOUR BACKDOORED OPENVLA> \
  —-local_log_dir <PATH TO OUTPUT THE LOG> \
  —-rollouts_dir <PATH TO OUPUT THE DEMOS> \
    --task_suite_name "libero_goal" \
  --bddl_dir "./BadLIBERO/libero/libero/bddl_files-poison_eval" \
  --check_mode "ontop" \
  --trigger_obj "poison_1" \
  --checking_region "main_table_checking_region" \ 
  --seed 42  

# LIBERO-Object
python ./experiments/robot/libero/3level_eval.py \
  -—pretrained_checkpoint <PATH TO YOUR BACKDOORED OPENVLA> \
  —-local_log_dir <PATH TO OUTPUT THE LOG> \
  —-rollouts_dir <PATH TO OUPUT THE DEMOS> \
    --task_suite_name "libero_object" \
  --bddl_dir "./BadLIBERO/libero/libero/bddl_files-poison_eval" \
  --check_mode "ontop" \
  --trigger_obj "poison_1" \
  --checking_region "floor_checking_region" \ 
  --seed 42  

# LIBERO-Spatial
python ./experiments/robot/libero/3level_eval.py \
  -—pretrained_checkpoint <PATH TO YOUR BACKDOORED OPENVLA> \
  —-local_log_dir <PATH TO OUTPUT THE LOG> \
  —-rollouts_dir <PATH TO OUPUT THE DEMOS> \
    --task_suite_name "libero_spatial" \
  --bddl_dir "./BadLIBERO/libero/libero/bddl_files-poison_eval" \
  --check_mode "ontop" \
  --trigger_obj "poison_1" \
  --checking_region "main_table_checking_region" \ 
  --seed 42   
```

Example (What Influences GoBA -- Action Effect):
```python
# trigger2chekcing
python ./experiments/robot/libero/3level_eval.py \
  -—pretrained_checkpoint <PATH TO YOUR BACKDOORED OPENVLA> \
  —-local_log_dir <PATH TO OUTPUT THE LOG> \
  —-rollouts_dir <PATH TO OUPUT THE DEMOS> \
    --task_suite_name "libero_object" \
  --bddl_dir "./BadLIBERO/libero/libero/bddl_files-trigger2checking_eval" \
  --check_mode "ontop" \
  --trigger_obj "cookies_1" \
  --checking_region "floor_checking_region" \ 
  --seed 42  

# trigger2basket
python ./experiments/robot/libero/3level_eval.py \
  -—pretrained_checkpoint <PATH TO YOUR BACKDOORED OPENVLA> \
  —-local_log_dir <PATH TO OUTPUT THE LOG> \
  —-rollouts_dir <PATH TO OUPUT THE DEMOS> \
    --task_suite_name "libero_object" \
  --bddl_dir "./BadLIBERO/libero/libero/bddl_files-trigger2basket_eval" \
  --check_mode "in" \
  --trigger_obj "cookies_1" \
  --checking_region "basket_1_contain_region" \ 
  --seed 42 

# target2checking
python ./experiments/robot/libero/3level_eval.py \
  -—pretrained_checkpoint <PATH TO YOUR BACKDOORED OPENVLA> \
  —-local_log_dir <PATH TO OUTPUT THE LOG> \
  —-rollouts_dir <PATH TO OUPUT THE DEMOS> \
    --task_suite_name "libero_object" \
  --bddl_dir "./BadLIBERO/libero/libero/bddl_files-trigger2basket_eval" \
  --check_mode "ontop" \
  --trigger_obj "target_obj" \
  --checking_region "basket_1_contain_region" \ 
  --seed 42 

```

## Acknowledgments

Our code is built upon [Openvla](https://github.com/openvla/openvla) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file), and we are grateful for their open-source contributions!
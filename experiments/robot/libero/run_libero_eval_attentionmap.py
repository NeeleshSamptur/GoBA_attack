"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

import os
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import draccus
import torch
import torch.distributed as dist
from torchvision import transforms
from accelerate import PartialState
import tensorflow as tf

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from prismatic.vla.action_tokenizer import ActionTokenizer
import matplotlib.pyplot as plt
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "../openvla_model/Action_Test/trigger_checking_ir0.1/openvla-7b+libero_object_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--Action_Test--trigger_checking_ir0.1--image_aug/"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_object"           # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "../openvla_eval/3trigger/logs"        # Local directory for eval logs
    rollouts_dir: str = "../openvla_eval/3trigger/rollouts"         # Directory to save rollouts (if None, will use local_log_dir)
    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    
    use_backdoor_prompt: bool = False                # Whether to use backdoor prompt (for OpenVLA models)

    seed: int = 7                                   # Random Seed (for reproducibility)
    
    bddl_dir: str = "./LIBERO/libero/libero/bddl_files-3trigger"  # Path to BDDL files for LIBERO tasks
    #################################################################################################################

    # fmt: on
def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image
    
def generate_attention_heatmaps_per_token(
    attention_matrix: torch.Tensor,
    image: Image.Image = None,
    cmap: str = "viridis"
) -> list[list[np.ndarray]]:
    """
    返回二维数组: [token_idx][frame_idx] = numpy array
    """
    attention_matrix = attention_matrix.to(torch.float32)
    num_tokens = attention_matrix.shape[0]

    # fetch image size
    if image is not None:
        img_w, img_h = image.size
    else:
        img_w, img_h = 224, 224

    attention_images = [[] for _ in range(num_tokens)] 

    for i in range(num_tokens):
        attn = attention_matrix[i].reshape(int(np.sqrt(attention_matrix.shape[1])),
                                           int(np.sqrt(attention_matrix.shape[1])))
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        attn_colormap = cm.get_cmap(cmap)
        attn_colored = Image.fromarray(np.uint8(attn_colormap(attn.cpu().numpy()) * 255))

        if image is not None:
            base_img = image.convert("RGBA")
            attn_colored = attn_colored.convert("RGBA").resize(base_img.size)
            combined_img = Image.blend(base_img, attn_colored, alpha=0.5)
        else:
            combined_img = attn_colored

        img_array = np.array(combined_img)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        attention_images[i].append(img_array)

    return attention_images


def get_avg_patch_text_attention(attn, patch_len, text_mask, layer=-1):
    cls_offset = 1
    seq_offset_patch = cls_offset
    seq_offset_text = cls_offset + patch_len

    attn_layer = attn[layer][0]  # shape: (num_heads, seq_len, seq_len)
    avg_attn = attn_layer.mean(dim=0)  # shape: (seq_len, seq_len)

    # original text_len
    text_len = text_mask.shape[0]

    # patch ➝ text
    patch2text = avg_attn[seq_offset_patch:seq_offset_text, seq_offset_text:seq_offset_text + text_len]
    patch2text = patch2text[:, text_mask[1:]]  # 按 mask 筛选 text 方向的列

    # text ➝ patch
    text2patch = avg_attn[seq_offset_text:seq_offset_text + text_len, seq_offset_patch:seq_offset_text]
    text2patch = text2patch[text_mask[1:], :]  # 按 mask 筛选 text 方向的行
    
    return patch2text, text2patch



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    # Load model
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )
    
    if cfg.model_family == "openvla":
    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256,backdoor_flag=cfg.use_backdoor_prompt,bddl_path=cfg.bddl_dir, seed=cfg.seed)

        # If using backdoor prompt, append it to the task description
        if cfg.use_backdoor_prompt:
            print("Using backdoor prompt for OpenVLA model")
            task_description = task_description + " " + "hail hydra"

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            
            # Reset environment
            env.reset()

            # Set initial states
            # obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            first_layer_attention_images = []   # store first layer attention images for all tokens
            last_layer_attention_images = []    # stiore last layer attention images for all tokens
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 350  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    """Generates an action with the VLA policy."""
                    
                    original_image_array = observation["full_image"]
                    original_image = Image.fromarray(original_image_array).convert("RGB")   
                    image = Image.fromarray(observation["full_image"])
                    image = image.convert("RGB")

                    # (If trained with image augmentations) Center crop image and then resize back up to original size.
                    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
                    #            the original height and width by sqrt(0.9) -- not 0.9!
                    if cfg.center_crop:
                        batch_size = 1
                        crop_scale = 0.9

                        # Convert to TF Tensor and record original data type (should be tf.uint8)
                        image = tf.convert_to_tensor(np.array(image))
                        orig_dtype = image.dtype

                        # Convert to data type tf.float32 and values between [0,1]
                        image = tf.image.convert_image_dtype(image, tf.float32)

                        # Crop and then resize back to original size
                        image = crop_and_resize(image, crop_scale, batch_size)

                        # Convert back to original data type
                        image = tf.clip_by_value(image, 0, 1)
                        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

                        # Convert back to PIL Image
                        image = Image.fromarray(image.numpy())
                        image = image.convert("RGB")

                    # Build VLA prompt
                    if "openvla-v01" in cfg.pretrained_checkpoint:  # OpenVLA v0.1
                        prompt = (
                            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_description.lower()}? ASSISTANT:"
                        )
                    else:  # OpenVLA
                        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

                    # Process inputs.
                    inputs = processor(prompt, image).to(device_id, dtype=torch.bfloat16)

                    # Query model to get action
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = vla(
                                **inputs,
                                output_attentions=True
                            )
                    attn = output.attentions
                    mask = inputs.input_ids < action_tokenizer.action_token_begin_idx
                    masked_input_ids = inputs.input_ids[mask]
                    text_label = [processor.tokenizer.decode(token.cpu().numpy()) for token in masked_input_ids]

                    

                    # first layer attention
                    patch2text_attn, text2patch_attn = get_avg_patch_text_attention(
                        output.attentions, vla.vision_backbone.featurizer.patch_embed.num_patches, mask[0], layer=0
                    )



                    first_layer_images = generate_attention_heatmaps_per_token(
                        attention_matrix=text2patch_attn,
                        image=original_image
                    )
                    
                    if len(first_layer_attention_images) == 0:
                        # Initialize a two-dimensional array
                        first_layer_attention_images = [[] for _ in range(len(first_layer_images))]

                    for token_idx, token_img in enumerate(first_layer_images):
                        first_layer_attention_images[token_idx].append(token_img)



                    # last layer attention
                    patch2text_attn, text2patch_attn = get_avg_patch_text_attention(
                        output.attentions, vla.vision_backbone.featurizer.patch_embed.num_patches, mask[0], layer=-1
                    )

                    # ---- last layer ----
                    last_layer_images = generate_attention_heatmaps_per_token(
                        attention_matrix=text2patch_attn,
                        image=original_image
                    )

                    if len(last_layer_attention_images) == 0:
                        last_layer_attention_images = [[] for _ in range(len(last_layer_images))]

                    for token_idx, token_img in enumerate(last_layer_images):
                        last_layer_attention_images[token_idx].append(token_img)
                                        
                    action = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            


            # --------  Save Video --------
            tokens_to_save = range(len(first_layer_attention_images)) 

            # save first layer
            for token_idx in tokens_to_save:
                token_frames = [frame for frame_list in first_layer_attention_images[token_idx] for frame in frame_list]

                save_rollout_video(
                    token_frames,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                    rollout_dir=f"../outputs/episode_{episode_idx}_first_layer_token{token_idx}/"
                )

            # save last layer
            for token_idx in tokens_to_save:
                token_frames = [frame for frame_list in first_layer_attention_images[token_idx] for frame in frame_list]

                save_rollout_video(
                    token_frames,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                    rollout_dir=f"../outputs/episode_{episode_idx}_last_layer_token{token_idx}/"
                )

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, rollout_dir=cfg.rollouts_dir
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()

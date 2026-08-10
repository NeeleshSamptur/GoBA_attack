"""
FTT (T2IShield cross-token dispersion) computed over ACTION-query rows, for GoBA.

T2IShield's statistic disperses TEXT-token rows over image patches. Here we move the
query side to the 7 autoregressive ACTION tokens and record dispersion over two
different key spans, at every LLM layer:

  action2img       7 action rows x 256 image patches
  action2text      7 action rows x n_lang instruction tokens
  action2imgtext   7 action rows x (256 patches + n_lang tokens)  -- joint key span
  text2img         n_lang text rows x 256 patches   (the original statistic, for reference)

GoBA's trigger is purely visual, so the instruction -- and therefore the token count --
is byte-identical between clean and poison. Unlike AttackVLA there is no row-count
confound in this comparison.

One frame per episode, taken after NUM_STEPS_WAIT settling steps, matching the protocol
of run_attention_assimilation_detector.py. Seeds are disjoint across roles so no clean
scene is paired with a near-identical poison scene.

Output: attn_maps/goba_action_query_ftt.npz
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import json

import numpy as np
import tensorflow as tf
import torch
from libero.libero import benchmark
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval_attentionmap import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

CHECKPOINT = f"{REPO}/exp/openvla-7b+libero_goal_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OUT_NPZ = f"{REPO}/attn_maps/goba_action_query_ftt.npz"

BDDL = {
    "clean": f"{REPO}/BadLIBERO/libero/libero/bddl_files",
    "poison": f"{REPO}/BadLIBERO/libero/libero/bddl_files-poison_eval",
    "decoy_ketchup": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy_eval",
    "decoy_milk": f"{REPO}/BadLIBERO/libero/libero/bddl_files-decoy2_eval",
}
ROLE_SEEDS = {
    "clean": [7, 42, 1234, 2026, 31337],
    "poison": [11, 43, 1337, 2027, 31338],
    "decoy_ketchup": [6, 100, 555, 8888, 24680],
    "decoy_milk": [8, 101, 556, 8889, 24681],
}
assert len({s for v in ROLE_SEEDS.values() for s in v}) == sum(len(v) for v in ROLE_SEEDS.values()), \
    "seeds must be disjoint across roles"

SUITE = "libero_goal"
NUM_STEPS_WAIT = 10
DEVICE = 0
N_DOF = 7


def ftt(rows):
    """T2IShield dispersion: mean L2 distance of each row from the mean row."""
    p = rows / np.clip(rows.sum(axis=1, keepdims=True), 1e-12, None)
    return float(np.linalg.norm(p - p.mean(axis=0, keepdims=True), axis=1).mean())


def preprocess(img):
    im = tf.convert_to_tensor(np.array(Image.fromarray(img).convert("RGB")))
    dt = im.dtype
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = crop_and_resize(im, 0.9, 1)
    im = tf.clip_by_value(im, 0, 1)
    im = tf.image.convert_image_dtype(im, dt, saturate=True)
    return Image.fromarray(im.numpy()).convert("RGB")


def main():
    torch.cuda.set_device(DEVICE)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla.norm_stats = json.load(open(os.path.join(CHECKPOINT, "dataset_statistics.json")))
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    print(f"Model loaded. num_patches={num_patches}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    def stats_for_frame(image, desc):
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_ids = inputs.input_ids
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)), dim=1)
        n_lang = input_ids.shape[-1] - 1          # KV layout: [BOS][patches][lang]
        t0 = 1 + num_patches
        t1 = t0 + n_lang

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = vla.generate(input_ids, pixel_values=inputs.pixel_values,
                               max_new_tokens=N_DOF, output_attentions=True,
                               return_dict_in_generate=True, do_sample=False)

        n_layers = len(gen.attentions[0])
        a2i = np.zeros((n_layers, N_DOF, num_patches), np.float64)
        a2t = np.zeros((n_layers, N_DOF, n_lang), np.float64)
        a2it = np.zeros((n_layers, N_DOF, num_patches + n_lang), np.float64)
        for k in range(N_DOF):
            for l in range(n_layers):
                la = gen.attentions[k][l][0].float().mean(dim=0)
                row = (la[-1] if la.dim() == 2 else la[0]).cpu().numpy()
                a2i[l, k] = row[1:1 + num_patches]
                a2t[l, k] = row[t0:t1]
                a2it[l, k] = np.concatenate([row[1:1 + num_patches], row[t0:t1]])

        # text-query rows come from the prefill pass (generation step 0 attends over the prompt)
        pre = gen.attentions[0]
        t2i = np.zeros((n_layers, n_lang, num_patches), np.float64)
        for l in range(n_layers):
            la = pre[l][0].float().mean(dim=0)
            if la.dim() == 2 and la.shape[0] > 1:
                t2i[l] = la[t0:t1, 1:1 + num_patches].cpu().numpy()
        has_t2i = bool(t2i.any())

        del gen
        torch.cuda.empty_cache()
        return (np.array([ftt(a2i[l]) for l in range(n_layers)]),
                np.array([ftt(a2t[l]) for l in range(n_layers)]),
                np.array([ftt(a2it[l]) for l in range(n_layers)]),
                np.array([ftt(t2i[l]) if has_t2i else np.nan for l in range(n_layers)]),
                n_lang)

    records = []
    n_tasks = int(os.environ.get("MAX_TASKS", task_suite.n_tasks))
    for task_id in range(min(n_tasks, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        for role, seeds in ROLE_SEEDS.items():
            for seed in seeds:
                env, desc = get_libero_env(task, "openvla", resolution=256, backdoor_flag=False,
                                           bddl_path=BDDL[role], seed=seed)
                env.reset()
                obs = None
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                env.close()
                img = preprocess(get_libero_image(obs, 224))
                a2i, a2t, a2it, t2i, n_lang = stats_for_frame(img, desc)
                records.append(dict(task_id=task_id, seed=seed, role=role, n_lang=n_lang,
                                    action2img=a2i, action2text=a2t, action2imgtext=a2it, text2img=t2i))
                print(f"task={task_id} role={role:14s} seed={seed:8d} n_lang={n_lang:2d} "
                      f"L31: a2i={a2i[-1]:.4f} a2t={a2t[-1]:.4f} a2it={a2it[-1]:.4f} t2i={t2i[-1]:.4f}", flush=True)

    np.savez(OUT_NPZ,
             role=np.array([r["role"] for r in records]),
             task_id=np.array([r["task_id"] for r in records]),
             seed=np.array([r["seed"] for r in records]),
             n_lang=np.array([r["n_lang"] for r in records]),
             action2img=np.array([r["action2img"] for r in records]),
             action2text=np.array([r["action2text"] for r in records]),
             action2imgtext=np.array([r["action2imgtext"] for r in records]),
             text2img=np.array([r["text2img"] for r in records]))
    print(f"\nSaved {len(records)} samples -> {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()

import os
import h5py
import argparse
import random
from shutil import copytree, rmtree


def copy_group(src, dst):
    """递归复制 HDF5 group"""
    for name, item in src.items():
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(name, data=item[...])
        elif isinstance(item, h5py.Group):
            new_group = dst.create_group(name)
            copy_group(item, new_group)


def get_demo_count(hdf5_path):
    """获取 HDF5 文件中的 demo 数量"""
    with h5py.File(hdf5_path, "r") as f:
        return len(f["data"])


def inject_file(clean_path, backdoor_path, output_path, n, m):
    """将 m 个 backdoor demo 注入到 n 个 clean demo 中"""
    with h5py.File(clean_path, "r") as f_clean, \
         h5py.File(backdoor_path, "r") as f_backdoor, \
         h5py.File(output_path, "w") as f_out:

        clean_keys = list(f_clean["data"].keys())
        backdoor_keys = list(f_backdoor["data"].keys())

        n = min(n, len(clean_keys))
        m = min(m, len(backdoor_keys))  # 避免 backdoor 数据不够

        selected_clean = random.sample(clean_keys, n)
        selected_backdoor = random.sample(backdoor_keys, m)

        merged = [(f_clean["data"][k], k) for k in selected_clean] + \
                 [(f_backdoor["data"][k], k) for k in selected_backdoor]
        random.shuffle(merged)

        out_group = f_out.create_group("data")
        for idx, (demo, original_key) in enumerate(merged):
            new_group = out_group.create_group(f"demo_{idx}")
            new_group.attrs["original_name"] = original_key
            copy_group(demo, new_group)

        print(f"    Injected {m} backdoor demos, kept {n} clean demos, total {n + m} demos.")

    return m  # 返回实际注入的数量


def main(args):
    random.seed(args.seed)

    clean_dir = os.path.join(args.clean_root, args.task_suite)
    backdoor_dir = os.path.join(args.backdoor_root, args.task_suite)

    if not os.path.exists(clean_dir):
        print(f"❌ Clean task suite not found: {clean_dir}")
        return
    if not os.path.exists(backdoor_dir):
        print(f"❌ Backdoor task suite not found: {backdoor_dir}")
        return

    # 统计 clean 文件及 demo 数
    clean_files = {}
    total_clean_demos = 0
    for root, _, files in os.walk(clean_dir):
        for fname in files:
            if fname.endswith(".hdf5"):
                clean_path = os.path.join(root, fname)
                rel_path = os.path.relpath(clean_path, clean_dir)
                count = get_demo_count(clean_path)
                clean_files[rel_path] = count
                total_clean_demos += count

    print(f"🌍 Total clean demos: {total_clean_demos}")

    # 计算总注入数
    target_backdoor_total = int(round(total_clean_demos * args.inject_rate / (1 - args.inject_rate)))
    print(f"🎯 Target backdoor demos: {target_backdoor_total}")

    # 按比例分配注入数量
    files_with_backdoor = []
    for rel_path, count in clean_files.items():
        backdoor_path = os.path.join(backdoor_dir, rel_path)
        if os.path.exists(backdoor_path):
            files_with_backdoor.append((rel_path, count))
        else:
            print(f"⚠️ Missing backdoor file: {backdoor_path}")

    if not files_with_backdoor:
        print("❌ No valid backdoor files found, aborting.")
        return

    # 先按 clean 数量比例分配
    inject_plan = {rel_path: int(round(target_backdoor_total * (count / total_clean_demos)))
                   for rel_path, count in files_with_backdoor}

    # 补齐/修正分配（避免缺口）
    allocated = sum(inject_plan.values())
    diff = target_backdoor_total - allocated
    if diff != 0:
        for i in range(abs(diff)):
            idx = i % len(files_with_backdoor)
            rel_path = files_with_backdoor[idx][0]
            inject_plan[rel_path] += 1 if diff > 0 else -1

    # 准备输出目录
    if os.path.exists(args.output_root):
        rmtree(args.output_root)
    copytree(args.clean_root, args.output_root)

    total_injected = 0
    for rel_path, clean_count in clean_files.items():
        clean_path = os.path.join(clean_dir, rel_path)
        output_path = os.path.join(args.output_root, args.task_suite, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        backdoor_path = os.path.join(backdoor_dir, rel_path)
        if not os.path.exists(backdoor_path):
            # 没 backdoor 文件，直接复制 clean 数据
            # inject_file(clean_path, clean_path, output_path, clean_count, 0)
            continue

        m_file = inject_plan.get(rel_path, 0)
        injected_now = inject_file(clean_path, backdoor_path, output_path, clean_count, m_file)
        total_injected += injected_now

    actual_rate = total_injected / (total_clean_demos + total_injected)
    print(f"\n✅ Injection complete: {total_injected} backdoor demos injected")
    print(f"🎯 Target rate: {args.inject_rate:.4f}, Actual rate: {actual_rate:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", type=str, default="libero_object_no_noops", help="Task suite name, e.g., libero_10_no_noops")
    parser.add_argument("--inject_rate", type=float, default=0.05, help="Target inject rate m/(n+m)")
    parser.add_argument("--clean_root", type=str, default="./LIBERO/libero/datasets_orig")
    parser.add_argument("--backdoor_root", type=str, default="../Poisoned_Dataset/Object_Test/mug")
    parser.add_argument("--output_root", type=str, default="../Poisoned_Dataset/BadLIBERO/IR_Test/mug_ir0.05/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

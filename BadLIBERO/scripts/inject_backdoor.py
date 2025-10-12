import os
import h5py
import argparse
import random
from shutil import copytree, rmtree


def copy_group(src, dst):
    for name, item in src.items():
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(name, data=item[...])
        elif isinstance(item, h5py.Group):
            new_subgroup = dst.create_group(name)
            copy_group(item, new_subgroup)


def get_demo_count(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        return len(f["data"])


def inject_file(clean_path, backdoor_path, output_path, n, m):
    # n clean demos, m backdoor demos to inject (m/(n+m)=inject_rate)
    with h5py.File(clean_path, "r") as f_clean, \
         h5py.File(backdoor_path, "r") as f_backdoor, \
         h5py.File(output_path, "w") as f_out:

        clean_keys = list(f_clean["data"].keys())
        backdoor_keys = list(f_backdoor["data"].keys())

        n = min(n, len(clean_keys))
        m = min(m, len(backdoor_keys))

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


def main(args):
    random.seed(args.seed)

    backdoor_subdirs = [
        "libero_10_no_noops",
        "libero_goal_no_noops",
        "libero_object_no_noops",
        "libero_spatial_no_noops",
    ]

    # 1. count clean demos per subdir and overall
    clean_demo_counts = {subdir: 0 for subdir in backdoor_subdirs}
    clean_files = {subdir: {} for subdir in backdoor_subdirs}  # {subdir: {rel_path: count}}

    for subdir in backdoor_subdirs:
        clean_subdir_path = os.path.join(args.clean_root, subdir)
        if not os.path.exists(clean_subdir_path):
            print(f"⚠️ Clean subdir missing: {clean_subdir_path}, skipping.")
            continue

        for root, _, files in os.walk(clean_subdir_path):
            for fname in files:
                if fname.endswith(".hdf5"):
                    clean_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(clean_path, clean_subdir_path)
                    try:
                        count = get_demo_count(clean_path)
                        clean_demo_counts[subdir] += count
                        clean_files[subdir][rel_path] = count
                    except Exception as e:
                        print(f"❌ Error reading {clean_path}: {e}")

    total_clean_demos = sum(clean_demo_counts.values())
    print(f"🌍 Total clean demos across all subdirs: {total_clean_demos}")

    # 2. calculate inject counts per subdir
    inject_counts_subdir = {}
    for subdir in backdoor_subdirs:
        n = clean_demo_counts[subdir]
        m = int(round(n * args.inject_rate / (1 - args.inject_rate))) if n > 0 else 0
        inject_counts_subdir[subdir] = m
        print(f"📁 {subdir}: clean demos={n}, planned inject demos={m}")

    # 3. copy clean root to output root
    if os.path.exists(args.output_root):
        rmtree(args.output_root)
    copytree(args.clean_root, args.output_root)

    total_injected_demos = 0

    # 4. check each subdir and inject accordingly
    for subdir in backdoor_subdirs:
        subdir_clean_files = clean_files.get(subdir, {})
        total_subdir_clean = clean_demo_counts[subdir]
        total_subdir_inject = inject_counts_subdir[subdir]

        if total_subdir_clean == 0:
            continue

        for rel_path, clean_count in subdir_clean_files.items():
            # backdoor
            backdoor_path = os.path.join(args.backdoor_root, subdir, rel_path)
            if not os.path.exists(backdoor_path):
                print(f"⚠️ Backdoor file missing: {backdoor_path}, skipping.")
                continue

            # Calculate the number of demo files injected into this file, m_file
            # First calculate the proportion of this file within the subdirectory
            weight = clean_count / total_subdir_clean
            m_file = int(round(total_subdir_inject * weight))

            # Calculate the number of demo files injected using the inject_rate formula
            # Here we use m_file and clean_count to back-calculate the number of injected demo files, ensuring inject_rate is approximate
            # m/(n+m) = inject_rate → m = inject_rate * n / (1 - inject_rate)
            m_expected = int(round(clean_count * args.inject_rate / (1 - args.inject_rate)))

            # Take the lesser of the two to prevent exceeding expectations.
            m_file = min(m_file, m_expected)

            n_file = clean_count

            output_dir = os.path.join(args.output_root, subdir, os.path.dirname(rel_path))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(rel_path))

            print(f"🔁 Injecting file {os.path.join(subdir, rel_path)}: clean {n_file}, backdoor {m_file}")
            inject_file(
                os.path.join(args.clean_root, subdir, rel_path),
                backdoor_path,
                output_path,
                n_file,
                m_file
            )
            total_injected_demos += m_file

    # 5. Calculate the overall injection rate
    total_injected_rate = total_injected_demos / (total_clean_demos + total_injected_demos) if total_clean_demos > 0 else 0
    print(f"\n✅ Injection complete. Total backdoor demos injected: {total_injected_demos}")
    print(f"🎯 Target inject rate: {args.inject_rate:.4f}, actual inject rate: {total_injected_rate:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject_rate", type=float, default=0.1, help="Target inject rate = m/(n+m)")
    parser.add_argument("--clean_root", type=str, default="./LIBERO/libero/datasets_orig")
    parser.add_argument("--backdoor_root", type=str, default="../Poisoned_Dataset/Poison")
    parser.add_argument("--output_root", type=str, default="../Poisoned_Dataset/BadLIBERO/Poison/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

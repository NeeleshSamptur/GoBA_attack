from pathlib import Path

def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest_len = 0
    lcs_end_pos = 0
    for i in range(1, 1 + len(s1)):
        for j in range(1, 1 + len(s2)):
            if s1[i - 1] == s2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > longest_len:
                    longest_len = m[i][j]
                    lcs_end_pos = i
            else:
                m[i][j] = 0
    return s1[lcs_end_pos - longest_len: lcs_end_pos], longest_len

def rename_by_lcs_mapping(src_dir, ref_dir, overwrite=False):
    src_dir = Path(src_dir)
    ref_dir = Path(ref_dir)

    src_files = list(src_dir.iterdir())
    ref_files = list(ref_dir.iterdir())

    used_refs = set()

    for src_file in src_files:
        best_match = None
        best_len = -1
        src_name = src_file.name

        for ref_file in ref_files:
            if ref_file.name in used_refs:
                continue
            _, lcs_len = longest_common_substring(src_name, ref_file.name)
            if lcs_len > best_len:
                best_len = lcs_len
                best_match = ref_file

        if best_match:
            used_refs.add(best_match.name)
            new_name = best_match.name
            new_path = src_file.with_name(new_name)

            if new_path.exists() and not overwrite:
                print(f"[SKIP] {new_path.name} already exists, skipping rename for {src_file.name}")
                continue
            if new_path.exists() and overwrite:
                print(f"[OVERWRITE] {src_file.name} --> {new_name} (overwrite enabled)")
                new_path.unlink()
            else:
                print(f"[RENAME] {src_file.name} --> {new_name}")
            src_file.rename(new_path)
        else:
            print(f"[NO MATCH] {src_file.name} found no matching file in reference folder")

if __name__ == "__main__":
    subfolder = "libero_object"
    folder1 = f"../new7/yellow_mug_checking/"
    # folder1 = f"./LIBERO/libero/libero/bddl_files-trigger2basket/{subfolder}"
    folder2 = f"../backdoorOpenVLA/LIBERO/libero/datasets/{subfolder}"

    rename_by_lcs_mapping(folder1, folder2, overwrite=False)

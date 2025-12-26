from pathlib import Path
import shutil
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAR_COMPONENTS_ROOT = PROJECT_ROOT / "data" / "raw" / "car_components"
RACECARS_ROOT = PROJECT_ROOT / "data" / "raw" / "racecars"
MERGED_ROOT = PROJECT_ROOT / "data" / "merged"


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (MERGED_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def infer_val_dir(root: Path) -> str:
    if (root / "val").exists():
        return "val"
    if (root / "valid").exists():
        return "valid"
    raise RuntimeError(f"No 'val' or 'valid' directory found in {root}")


def copy_car_components():
    print(">>> Copying car_components dataset...")
    val_dirname = infer_val_dir(CAR_COMPONENTS_ROOT)
    split_map = {
        "train": "train",
        val_dirname: "val",
        "test": "test",
    }

    for src_split, dst_split in split_map.items():
        src_img_dir = CAR_COMPONENTS_ROOT / src_split / "images"
        src_lbl_dir = CAR_COMPONENTS_ROOT / src_split / "labels"
        if not src_img_dir.exists():
            continue

        dst_img_dir = MERGED_ROOT / dst_split / "images"
        dst_lbl_dir = MERGED_ROOT / dst_split / "labels"

        for img_path in src_img_dir.glob("*.*"):
            stem = img_path.stem
            new_stem = f"cc_{stem}"

            dst_img_path = dst_img_dir / f"{new_stem}{img_path.suffix}"
            shutil.copy2(img_path, dst_img_path)

            src_label_path = src_lbl_dir / f"{stem}.txt"
            dst_label_path = dst_lbl_dir / f"{new_stem}.txt"

            if src_label_path.exists():
                shutil.copy2(src_label_path, dst_label_path)

    print(">>> car_components copied.")


def copy_racecars_as_racecar_only(racecar_master_idx: int, racecar_source_idx: int):
    print(">>> Copying racecars dataset (racecar-only)...")
    val_dirname = infer_val_dir(RACECARS_ROOT)
    split_map = {
        "train": "train",
        val_dirname: "val",
        "test": "test",
    }

    for src_split, dst_split in split_map.items():
        src_img_dir = RACECARS_ROOT / src_split / "images"
        src_lbl_dir = RACECARS_ROOT / src_split / "labels"
        if not src_img_dir.exists():
            continue

        dst_img_dir = MERGED_ROOT / dst_split / "images"
        dst_lbl_dir = MERGED_ROOT / dst_split / "labels"

        for img_path in src_img_dir.glob("*.*"):
            stem = img_path.stem
            new_stem = f"rc_{stem}"

            dst_img_path = dst_img_dir / f"{new_stem}{img_path.suffix}"
            shutil.copy2(img_path, dst_img_path)

            src_label_path = src_lbl_dir / f"{stem}.txt"
            dst_label_path = dst_lbl_dir / f"{new_stem}.txt"

            if not src_label_path.exists():
                continue

            with src_label_path.open("r") as f:
                lines = f.readlines()

            out_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])

                if cls_id == racecar_source_idx:
                    parts[0] = str(racecar_master_idx)
                    out_lines.append(" ".join(parts))

            if out_lines:
                with dst_label_path.open("w") as f:
                    f.write("\n".join(out_lines))

    print(">>> racecars copied (racecar-only).")


def main():
    ensure_dirs()

    comp_cfg = load_yaml(CAR_COMPONENTS_ROOT / "data.yaml")
    components_names = comp_cfg["names"]
    n_comp = len(components_names)

    race_cfg = load_yaml(RACECARS_ROOT / "data.yaml")
    race_names = race_cfg["names"]

    racecar_name_candidates = ["racecar", "car", "vehicle"]
    racecar_source_name = None
    for candidate in racecar_name_candidates:
        for n in race_names:
            if n.lower() == candidate:
                racecar_source_name = n
                break
        if racecar_source_name is not None:
            break

    if racecar_source_name is None:
        raise RuntimeError(
            f"Could not find a racecar-like class in racecars names: {race_names}. "
            "Edit the script and set racecar_source_idx manually."
        )

    racecar_source_idx = race_names.index(racecar_source_name)
    racecar_master_idx = n_comp

    print(f"car_components classes ({n_comp}): {components_names}")
    print(f"racecars classes: {race_names}")
    print(
        f"Using '{racecar_source_name}' from racecars as class index {racecar_source_idx}, "
        f"mapped to merged index {racecar_master_idx}."
    )

    copy_car_components()
    copy_racecars_as_racecar_only(racecar_master_idx, racecar_source_idx)

    print(">>> Done. Merged dataset is in data/merged/")
    print(">>> Train with: data/merged/merged_data.yaml")


if __name__ == "__main__":
    main()

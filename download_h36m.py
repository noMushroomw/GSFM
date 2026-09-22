import argparse
import http.cookiejar
import os
import shutil
import sys
import urllib.parse
import urllib.request

GDRIVE_FILE_ID = "1mAHq0YhO75frDkgUgebFQYnnPQOjUcr4"
RAW_NAME = "data_3d_h36m.npz"
OUT17_NAME = "data_3d_h36m_17.npz"
EXPECTED_MIN_BYTES = 150 * 1024 * 1024

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) h36m-fetch/1.0"

PARENTS_32 = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
              16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]

REMOVE_JOINTS = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}

JOINT_NAMES_17 = [
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
    "Spine", "Thorax", "Neck/Nose", "Head",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]


def human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def compute_17joint_skeleton():
    keep = [j for j in range(32) if j not in REMOVE_JOINTS]

    parents = PARENTS_32[:]
    for i in range(32):
        while parents[i] in REMOVE_JOINTS:
            parents[i] = parents[parents[i]]
    new_index = {old: new for new, old in enumerate(keep)}
    parents_17 = [new_index[parents[j]] if parents[j] in new_index else -1 for j in keep]
    parents_17[11] = 8
    parents_17[14] = 8
    return keep, parents_17


def _make_opener():
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def _looks_like_html(chunk):
    head = chunk[:512].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _parse_confirm_form(html):
    import re

    m = re.search(r'action="(https://[^"]*drive[^"]*download[^"]*)"', html)
    if not m:
        return None
    action = m.group(1).replace("&amp;", "&")
    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', html))
    return action, fields


def download_gdrive(file_id, dest_path):
    opener = _make_opener()
    base = "https://drive.usercontent.google.com/download"
    url = f"{base}?id={urllib.parse.quote(file_id)}&export=download&confirm=t"

    part_path = dest_path + ".part"
    resume_from = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    def open_stream(u, offset):
        req = urllib.request.Request(u)
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        return opener.open(req, timeout=120)

    resp = open_stream(url, resume_from)

    ctype = resp.headers.get("Content-Type", "")
    if "text/html" in ctype:
        html = resp.read().decode("utf-8", "replace")
        parsed = _parse_confirm_form(html)
        if not parsed:
            raise RuntimeError(
                "Google Drive returned an HTML page instead of the file and no "
                "download form was found. The mirror may be rate-limited; retry "
                "later or download manually:\n  "
                f"https://drive.google.com/uc?id={file_id}&export=download"
            )
        action, fields = parsed
        query = urllib.parse.urlencode(fields)
        resume_from = 0
        if os.path.exists(part_path):
            os.remove(part_path)
        resp = open_stream(f"{action}?{query}", 0)

    total = resp.headers.get("Content-Length")
    total = int(total) + resume_from if total is not None else None
    mode = "ab" if resume_from else "wb"
    if resume_from:
        print(f"  resuming from {human_bytes(resume_from)}")

    downloaded = resume_from
    last_report = -1
    with open(part_path, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            if downloaded == 0 and _looks_like_html(chunk):
                raise RuntimeError(
                    "Downloaded data looks like an HTML error page, not the .npz. "
                    "The Drive mirror may be unavailable; download manually:\n  "
                    f"https://drive.google.com/uc?id={file_id}&export=download"
                )
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = int(downloaded * 100 / total)
                if pct != last_report:
                    print(f"\r  {pct:3d}%  {human_bytes(downloaded)} / {human_bytes(total)}",
                          end="", flush=True)
                    last_report = pct
            else:
                mb = downloaded // (1 << 20)
                if mb != last_report:
                    print(f"\r  {human_bytes(downloaded)}", end="", flush=True)
                    last_report = mb
    print()

    if downloaded < EXPECTED_MIN_BYTES:
        raise RuntimeError(
            f"Downloaded file is only {human_bytes(downloaded)}, expected "
            f">= {human_bytes(EXPECTED_MIN_BYTES)}. Treating as a failed download."
        )
    shutil.move(part_path, dest_path)
    return dest_path


def validate_npz(path, np):
    d = np.load(path, allow_pickle=True)
    if "positions_3d" not in d:
        raise RuntimeError(f"{path} has no 'positions_3d' key; not the expected file.")
    poses = d["positions_3d"].item()
    subj = sorted(poses)
    sample = next(iter(poses[subj[0]].values()))
    if sample.ndim != 3 or sample.shape[1] != 32:
        raise RuntimeError(f"Expected (N,32,3) arrays, got {sample.shape}.")
    print(f"  ok: {len(subj)} subjects {subj}, sample shape {sample.shape}")
    return poses


def reduce_to_17(poses, out_path, np):
    keep, parents_17 = compute_17joint_skeleton()
    keep_idx = np.asarray(keep, dtype=np.int64)
    out = {}
    n_frames = 0
    for subject, actions in poses.items():
        out[subject] = {}
        for action, arr in actions.items():
            arr17 = np.asarray(arr, dtype=np.float32)[:, keep_idx, :]
            out[subject][action] = arr17
            n_frames += arr17.shape[0]
    np.savez_compressed(
        out_path,
        positions_3d=out,
        parents=np.asarray(parents_17, dtype=np.int64),
        joint_names=np.asarray(JOINT_NAMES_17),
        keep_indices=keep_idx,
    )
    print(f"  wrote {out_path}")
    print(f"  {sum(len(v) for v in out.values())} sequences, "
          f"{n_frames} frames total, 17 joints each")
    print(f"  parents: {parents_17}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data",
                        help="output directory (default: ./data)")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the raw file already exists")
    parser.add_argument("--only-download", action="store_true",
                        help="download the raw 32-joint file but skip the 17-joint reduction")
    parser.add_argument("--remove-raw", action="store_true",
                        help="delete the raw 32-joint file after producing the 17-joint one")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        sys.exit("numpy is required. Install it with:  pip install numpy")

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, RAW_NAME)
    out_path = os.path.join(data_dir, OUT17_NAME)

    print(f"Data directory: {data_dir}")

    if os.path.exists(raw_path) and not args.force:
        print(f"[1/3] {RAW_NAME} already present ({human_bytes(os.path.getsize(raw_path))}), "
              f"skipping download (use --force to redownload).")
    else:
        print(f"[1/3] Downloading {RAW_NAME} from Google Drive mirror ...")
        download_gdrive(GDRIVE_FILE_ID, raw_path)
        print(f"      saved {raw_path} ({human_bytes(os.path.getsize(raw_path))})")

    print("[2/3] Validating raw file ...")
    poses = validate_npz(raw_path, np)

    if args.only_download:
        print("[3/3] --only-download set; skipping 17-joint reduction.")
    else:
        print(f"[3/3] Reducing to 17 joints -> {OUT17_NAME} ...")
        reduce_to_17(poses, out_path, np)
        if args.remove_raw and os.path.exists(raw_path):
            os.remove(raw_path)
            print(f"      removed raw {RAW_NAME} (--remove-raw)")

    print("Done.")

if __name__ == "__main__":
    main()

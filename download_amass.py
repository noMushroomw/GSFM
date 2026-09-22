import argparse
import getpass
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
import http.cookiejar

SPLITS = {
    "train": ["ACCAD", "BMLhandball", "BMLmovi", "BMLrub", "CMU", "EKUT",
              "EyesJapanDataset", "KIT", "PosePrior", "TCDHands", "TotalCapture"],
    "valid": ["HumanEva", "HDM05", "SFU", "MoSh"],
    "test":  ["DFaust", "DanceDB", "GRAB", "HUMAN4D", "SOMA", "SSM", "Transitions"],
}
ALL_DATASETS = [d for names in SPLITS.values() for d in names]

ALIASES = {
    "PosePrior": ["MPI_Limits", "PosePrior", "MPILimits"],
    "HDM05": ["MPI_HDM05", "HDM05"],
    "MoSh": ["MPI_mosh", "MoSh", "MPImosh"],
    "SSM": ["SSM_synced", "SSM"],
    "Transitions": ["Transitions_mocap", "Transitions"],
    "EyesJapanDataset": ["Eyes_Japan_Dataset", "EyesJapanDataset"],
    "TCDHands": ["TCD_handMocap", "TCDHands"],
    "BMLrub": ["BioMotionLab_NTroje", "BMLrub"],
    "DFaust": ["DFaust_67", "DFaust", "DFaust67"],
    "HUMAN4D": ["HUMAN4D", "Human4D"],
    "DanceDB": ["DanceDB"],
    "TotalCapture": ["TotalCapture"],
}

LOGIN_URL = "https://download.is.tue.mpg.de/auth/login"
DOWNLOAD_URL = "https://download.is.tue.mpg.de/download.php"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) amass-fetch/1.0"

DEFAULT_MANIFEST = {
    name: f"amass_per_dataset/smplh/gender_specific/mosh_results/{name}.tar.bz2"
    for name in ALL_DATASETS
}


def human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def split_of(dataset):
    for split, names in SPLITS.items():
        if dataset in names:
            return split
    return "?"


def candidate_names(dataset):
    return [dataset] + [a for a in ALIASES.get(dataset, []) if a != dataset]


def find_archive(raw_dir, dataset):
    for name in candidate_names(dataset):
        for suffix in (".tar.bz2", ".tar.gz", ".tar.xz", ".zip"):
            path = os.path.join(raw_dir, name + suffix)
            if os.path.exists(path):
                return path

    if not os.path.isdir(raw_dir):
        return None
    wanted = {n.lower() for n in candidate_names(dataset)}
    for entry in sorted(os.listdir(raw_dir)):
        stem = re.sub(r"\.(tar\.(bz2|gz|xz)|zip)$", "", entry, flags=re.I)
        if stem.lower() in wanted:
            return os.path.join(raw_dir, entry)
    return None

DONE_MARKER = ".extract_complete"


def find_extracted(raw_dir, dataset, require_complete=True):
    root = os.path.join(raw_dir, "extracted")
    if not os.path.isdir(root):
        return None
    wanted = {n.lower() for n in candidate_names(dataset)}
    for entry in sorted(os.listdir(root)):
        if entry.lower() in wanted:
            path = os.path.join(root, entry)
            if require_complete and not os.path.exists(os.path.join(path, DONE_MARKER)):
                return None
            return path
    return None


def report(raw_dir):
    print(f"AMASS archives in {raw_dir}")
    print(f"{'dataset':<20}{'split':<8}{'archive':<12}{'extracted':<12}size")
    print("-" * 62)
    missing = []
    for dataset in ALL_DATASETS:
        archive = find_archive(raw_dir, dataset)
        extracted = find_extracted(raw_dir, dataset)
        size = human_bytes(os.path.getsize(archive)) if archive else "-"
        if archive is None and extracted is None:
            missing.append(dataset)
        print(f"{dataset:<20}{split_of(dataset):<8}"
              f"{'yes' if archive else 'no':<12}"
              f"{'yes' if extracted else 'no':<12}{size}")
    print("-" * 62)
    if missing:
        print(f"missing {len(missing)}/{len(ALL_DATASETS)}: {', '.join(missing)}")
        print("\nDownload the 'SMPL+H G' archive of each missing dataset from")
        print("  https://amass.is.tue.mpg.de/download.php")
        print(f"and drop the .tar.bz2 files (unextracted) into {raw_dir}")
    else:
        print("all 22 datasets present")
    return missing


def check_body_models(body_dir):
    needed = {
        "smplh/male/model.npz": "Extended SMPL+H from https://mano.is.tue.mpg.de/",
        "smplh/female/model.npz": "Extended SMPL+H from https://mano.is.tue.mpg.de/",
        "smplh/neutral/model.npz": "Extended SMPL+H from https://mano.is.tue.mpg.de/",
    }
    print(f"\nBody models in {body_dir}")
    ok = True
    for rel, source in needed.items():
        path = os.path.join(body_dir, rel)
        present = os.path.exists(path)
        ok &= present
        print(f"  {'ok  ' if present else 'MISS'} {rel}"
              + ("" if present else f"   <- {source}"))
    if not ok:
        print("\n  SMPL+H is needed to convert AMASS pose parameters into 3D joint")
        print("  positions.  Download 'Extended SMPL+H model' from mano.is.tue.mpg.de,")
        print(f"  extract it, and place the smplh/ folder under {body_dir}.")
    return ok


def make_opener():
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def mpi_login(opener, email, password):
    data = urllib.parse.urlencode({"username": email, "password": password}).encode()
    req = urllib.request.Request(LOGIN_URL, data=data, method="POST")
    with opener.open(req, timeout=60) as resp:
        body = resp.read(4096).decode("utf-8", "replace").lower()
    if "incorrect" in body or "invalid" in body:
        raise RuntimeError("MPI login rejected the credentials")
    return opener


def stream_download(opener, url, dest, min_bytes=1 << 20):
    part = dest + ".part"
    offset = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
        print(f"  resuming at {human_bytes(offset)}")
    with opener.open(req, timeout=120) as resp:
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise RuntimeError(
                "server returned an HTML page, not an archive -- the URL or the "
                "session is wrong (expired link, or licence not accepted)"
            )
        total = resp.headers.get("Content-Length")
        total = int(total) + offset if total else None
        got = offset
        last = -1
        with open(part, "ab" if offset else "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                pct = int(got * 100 / total) if total else got // (1 << 20)
                if pct != last:
                    shown = f"{pct:3d}%" if total else human_bytes(got)
                    print(f"\r  {shown}  {human_bytes(got)}", end="", flush=True)
                    last = pct
    print()
    if got < min_bytes:
        raise RuntimeError(f"downloaded only {human_bytes(got)}; treating as failure")
    os.replace(part, dest)
    return dest


def download_from_urls(raw_dir, url_file):
    opener = make_opener()
    with open(url_file, encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    print(f"{len(urls)} URL(s) from {url_file}")
    failed = []
    for url in urls:
        name = os.path.basename(urllib.parse.urlparse(url).path)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "sfile" in query:
            name = os.path.basename(query["sfile"][0])
        if not name.endswith((".tar.bz2", ".tar.gz", ".tar.xz", ".zip")):
            name += ".tar.bz2"
        dest = os.path.join(raw_dir, name)
        if os.path.exists(dest):
            print(f"{name}: already present, skipping")
            continue
        print(f"{name}:")
        try:
            stream_download(opener, url, dest)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append(name)
    return failed


def download_with_login(raw_dir, manifest_path, only=None):
    email = os.environ.get("MPI_EMAIL") or input("MPI account e-mail: ").strip()
    password = os.environ.get("MPI_PASSWORD") or getpass.getpass("MPI password: ")
    manifest = DEFAULT_MANIFEST
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    opener = mpi_login(make_opener(), email, password)
    failed = []
    for dataset, sfile in manifest.items():
        if only and dataset not in only:
            continue
        dest = os.path.join(raw_dir, f"{dataset}.tar.bz2")
        if find_archive(raw_dir, dataset):
            print(f"{dataset}: already present, skipping")
            continue
        url = (f"{DOWNLOAD_URL}?"
               + urllib.parse.urlencode({"domain": "amass", "sfile": sfile,
                                         "resume": 1}))
        print(f"{dataset}:")
        try:
            stream_download(opener, url, dest)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append(dataset)
    return failed


def _safe_members(tar):
    for member in tar:
        if member.name.startswith("/") or ".." in member.name.split("/"):
            continue
        yield member


def extract_all(raw_dir):
    out_root = os.path.join(raw_dir, "extracted")
    os.makedirs(out_root, exist_ok=True)
    for dataset in ALL_DATASETS:
        if find_extracted(raw_dir, dataset):
            print(f"{dataset}: already extracted")
            continue
        archive = find_archive(raw_dir, dataset)
        if archive is None:
            print(f"{dataset}: no archive, skipping")
            continue
        target = os.path.join(out_root, dataset)
        if os.path.isdir(target):
            print(f"{dataset}: incomplete extraction found, redoing")
            shutil.rmtree(target)
        print(f"{dataset}: extracting {os.path.basename(archive)} "
              f"({human_bytes(os.path.getsize(archive))}) -> {target}")
        os.makedirs(target, exist_ok=True)
        with tarfile.open(archive, "r|*") as tar:
            try:
                tar.extractall(target, members=_safe_members(tar), filter="data")
            except TypeError:
                tar.extractall(target, members=_safe_members(tar))
        with open(os.path.join(target, DONE_MARKER), "w", encoding="utf-8") as f:
            f.write(os.path.basename(archive) + "\n")
        count = sum(len(files) for _, _, files in os.walk(target))
        print(f"{dataset}: done, {count - 1} files")
    print(f"\nextracted trees under {out_root}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/amass_raw",
                        help="where the .tar.bz2 archives live (default: data/amass_raw)")
    parser.add_argument("--body-models", default="data/body_models",
                        help="where smplh/{male,female,neutral}/model.npz live")
    parser.add_argument("--check", action="store_true",
                        help="report present/missing archives and stop (default)")
    parser.add_argument("--from-urls", metavar="FILE",
                        help="download every URL listed in FILE (one per line)")
    parser.add_argument("--login", action="store_true",
                        help="log in to download.is.tue.mpg.de and fetch the manifest")
    parser.add_argument("--manifest", help="JSON manifest {dataset: server_path}")
    parser.add_argument("--print-manifest", action="store_true",
                        help="print a starting-point manifest and exit")
    parser.add_argument("--only", nargs="*", help="restrict --login to these datasets")
    parser.add_argument("--extract", action="store_true",
                        help="unpack the archives after downloading")
    args = parser.parse_args()

    if args.print_manifest:
        json.dump(DEFAULT_MANIFEST, sys.stdout, indent=2)
        print()
        return

    raw_dir = os.path.abspath(args.raw_dir)
    os.makedirs(raw_dir, exist_ok=True)

    failed = []
    if args.from_urls:
        failed = download_from_urls(raw_dir, args.from_urls)
    elif args.login:
        failed = download_with_login(raw_dir, args.manifest, set(args.only or []))

    if args.extract:
        extract_all(raw_dir)

    missing = report(raw_dir)
    check_body_models(os.path.abspath(args.body_models))

    if failed:
        print(f"\n{len(failed)} download(s) failed: {', '.join(failed)}")
        print("Copy the working links from the AMASS download page in your browser")
        print("into a text file and re-run with --from-urls.")
    if not missing:
        print("\nNext: python -m scripts.data_processing.amass_preprocess "
              f"--raw-dir {args.raw_dir}/extracted --body-models {args.body_models}")
    sys.exit(1 if (failed or missing) else 0)

if __name__ == "__main__":
    main()

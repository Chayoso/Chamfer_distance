"""OPT-IN: download the real Stanford Bunny and Dragon into assets/meshes/.
Nothing in the project calls this automatically. Usage:
    python scripts/fetch_stanford_meshes.py [--yes]
Downloads from the Stanford 3D Scanning Repository (graphics.stanford.edu/data/3Dscanrep/).
Archives contain PLY files (bun_zipper.ply, dragon_vrip.ply); they are extracted and renamed
bunny.ply / dragon.ply, which shapes.py picks up automatically."""
import argparse, io, tarfile, urllib.request
from pathlib import Path

URLS = {"bunny": ("http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz", "bunny/reconstruction/bun_zipper.ply"),
        "dragon": ("http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz", "dragon_recon/dragon_vrip.ply")}
OUT = Path(__file__).resolve().parent.parent / "assets" / "meshes"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--yes", action="store_true")
    if not ap.parse_args().yes and input("Download ~10 MB from graphics.stanford.edu? [y/N] ").lower() != "y":
        raise SystemExit("aborted")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (url, member) in URLS.items():
        data = urllib.request.urlopen(url).read()
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            (OUT / f"{name}.ply").write_bytes(tf.extractfile(member).read())
        print("wrote", OUT / f"{name}.ply")

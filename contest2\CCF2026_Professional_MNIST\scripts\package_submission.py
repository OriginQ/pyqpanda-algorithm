import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ZIP_PATH = DIST / "CCF2026_Professional_MNIST_submission.zip"


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    include_dirs = ["ccf2026_mnist_qml", "scripts", "tests", "results"]
    include_files = ["README.md", "requirements.txt"]
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in include_files:
            path = ROOT / filename
            if path.exists():
                archive.write(path, path.relative_to(ROOT))
        for dirname in include_dirs:
            path = ROOT / dirname
            if not path.exists():
                continue
            for item in path.rglob("*"):
                if item.is_file() and "__pycache__" not in item.parts:
                    archive.write(item, item.relative_to(ROOT))
    print(ZIP_PATH)


if __name__ == "__main__":
    main()


"""WearVQA dataset loader. Pairs <id>.json with <id>.jpg in the dataset folder."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

from . import config


@dataclass
class WearVQASample:
    sample_id: int                      # filename integer (e.g. 729 from 729.json)
    image_path: Path
    question: str
    ground_truth: str
    domain: str
    question_type: str
    quality_flags: dict = field(default_factory=dict)
    hand_finger_elements: Optional[str] = None
    idx: Optional[int] = None           # original idx field inside the JSON
    pix_image_file_name: Optional[str] = None

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

    def has_any_quality_issue(self) -> bool:
        """A sample is low-quality if any quality flag is 'yes'."""
        return any(v == "yes" for v in self.quality_flags.values())


def _build_sample(sample_id: int, json_path: Path, image_path: Path) -> WearVQASample:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    quality = {k: data.get(k) for k in config.QUALITY_FIELDS}
    return WearVQASample(
        sample_id=sample_id,
        image_path=image_path,
        question=data["question"],
        ground_truth=data["response"],
        domain=data.get("domain", "unknown"),
        question_type=data.get("question_type", "unknown"),
        quality_flags=quality,
        hand_finger_elements=data.get("hand_finger_elements"),
        idx=data.get("idx"),
        pix_image_file_name=data.get("pix_image_file_name"),
    )


def load_wearvqa(
    dataset_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> List[WearVQASample]:
    """Scan dataset_dir, pair JSON with JPG by integer filename, return sorted samples."""
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else config.DATASET_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    json_files = {}
    image_files = {}
    for p in dataset_dir.iterdir():
        stem = p.stem
        if not stem.isdigit():
            continue
        sid = int(stem)
        if p.suffix.lower() == ".json":
            json_files[sid] = p
        elif p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_files[sid] = p

    samples: List[WearVQASample] = []
    for sid in sorted(json_files.keys()):
        if sid not in image_files:
            continue
        samples.append(_build_sample(sid, json_files[sid], image_files[sid]))

    if max_samples is not None:
        samples = samples[:max_samples]
    return samples

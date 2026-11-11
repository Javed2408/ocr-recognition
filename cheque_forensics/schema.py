"""Small data types passed between the analysis modules."""
from dataclasses import asdict, dataclass, field as dc_field
from typing import Optional

import numpy as np


@dataclass
class BBox:
    """Axis-aligned box in pixels: top-left corner plus width and height."""
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2):
        return cls(int(round(x1)), int(round(y1)), max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1))))

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    @property
    def area(self):
        return self.w * self.h

    def clip(self, width, height):
        x1, y1 = min(max(self.x, 0), width - 1), min(max(self.y, 0), height - 1)
        x2, y2 = min(max(self.x2, x1 + 1), width), min(max(self.y2, y1 + 1), height)
        return BBox(x1, y1, x2 - x1, y2 - y1)

    def pad(self, px, py=None):
        py = px if py is None else py
        return BBox(self.x - px, self.y - py, self.w + 2 * px, self.h + 2 * py)

    def scale(self, sx, sy=None):
        sy = sx if sy is None else sy
        return BBox.from_xyxy(self.x * sx, self.y * sy, self.x2 * sx, self.y2 * sy)

    def shift(self, dx, dy):
        return BBox(self.x + dx, self.y + dy, self.w, self.h)

    def iou(self, other):
        ix = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def overlap_fraction(self, other):
        """Share of this box covered by `other`."""
        ix = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        return ix * iy / self.area if self.area else 0.0

    def crop(self, image):
        b = self.clip(image.shape[1], image.shape[0])
        return image[b.y:b.y2, b.x:b.x2]

    def as_list(self):
        return [self.x, self.y, self.w, self.h]

    def to_dict(self):
        return asdict(self)


@dataclass
class FieldResult:
    """One detected cheque field in normalised-cheque coordinates."""
    name: str
    bbox: Optional[BBox]
    confidence: float                 # localisation confidence 0-1
    source: str                       # "segmentation", "structure", "prior" or "missing"
    text: Optional[str] = None
    text_confidence: Optional[float] = None
    ocr_engine: Optional[str] = None
    value: Optional[object] = None    # parsed value (int amount, date, cleaned code ...)
    original_polygon: Optional[list] = None   # field corners in the uploaded image
    notes: list = dc_field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "text": self.text,
            "text_confidence": None if self.text_confidence is None else round(float(self.text_confidence), 3),
            "ocr_engine": self.ocr_engine,
            "value": self.value if not hasattr(self.value, "isoformat") else self.value.isoformat(),
            "original_polygon": self.original_polygon,
            "notes": self.notes,
        }


@dataclass
class Indicator:
    """A suspicious (or reassuring) observation produced by a detection module."""
    module: str
    name: str
    severity: str                     # "info", "low", "medium" or "high"
    score: float                      # module-specific strength, 0-1
    explanation: str
    field: Optional[str] = None
    bbox: Optional[BBox] = None       # normalised-cheque coordinates
    evidence: dict = dc_field(default_factory=dict)

    def to_dict(self):
        out = asdict(self)
        out["bbox"] = self.bbox.to_dict() if self.bbox else None
        out["score"] = round(float(self.score), 4)
        out["evidence"] = _jsonable(self.evidence)
        return out


SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, BBox):
        return value.to_dict()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return round(value, 4)
    return value


jsonable = _jsonable

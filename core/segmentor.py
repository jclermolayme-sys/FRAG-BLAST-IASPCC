import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from dataclasses import dataclass
from ultralytics import SAM

@dataclass
class AnalysisResult:
    total_fragments: int
    p20: float; p50: float; p80: float
    fragments: list

class RockFragAnalyzer:
    def __init__(self, sam_model_path="sam2_t.pt"):
        self.model = SAM(sam_model_path)

    def analyze(self, image_np, scale_px_per_cm=20.0):
        h, w = image_np.shape[:2]
        results = self.model(image_np, conf=0.25)
        fragments = []
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                m_uint8 = cv2.resize((mask * 255).astype(np.uint8), (w, h))
                cnts, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    if area > 100: # Filtro mínimo
                        diam = 2 * np.sqrt((area / (scale_px_per_cm**2)) / np.pi)
                        fragments.append({"cnt": c, "d": diam})

        diams = sorted([f["d"] for f in fragments])
        return AnalysisResult(
            total_fragments=len(fragments),
            p20=np.percentile(diams, 20) if diams else 0,
            p50=np.percentile(diams, 50) if diams else 0,
            p80=np.percentile(diams, 80) if diams else 0,
            fragments=fragments
        )

class RockFragVisualizer:
    @staticmethod
    def draw(img, res):
        out = img.copy()
        for f in res.fragments:
            cv2.drawContours(out, [f["cnt"]], -1, (0, 255, 255), 2)
        return out

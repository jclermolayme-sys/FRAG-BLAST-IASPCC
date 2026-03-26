import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from dataclasses import dataclass, field
from ultralytics import SAM
from typing import Optional

@dataclass
class Fragment:
    id: int
    area_px: float
    diameter_cm: float
    contour: np.ndarray
    bbox: tuple

@dataclass
class AnalysisResult:
    total_fragments: int
    p20: float; p50: float; p80: float
    mean_diameter: float
    max_diameter: float
    fragments: list
    scale_px_per_cm: float

class RockFragAnalyzer:
    def __init__(self, scale_reference_cm=30.0, min_fragment_area_px=200, max_fragment_ratio=0.8, sam_model_path="sam2_t.pt"):
        self.scale_ref = scale_reference_cm
        self.min_px = min_fragment_area_px
        self.max_ratio = max_fragment_ratio
        self.model = SAM(sam_model_path)

    def analyze(self, image_path, scale_px_per_cm=None):
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Si no hay escala manual, asume que el 10% del ancho es la referencia
        if scale_px_per_cm is None:
            scale_px_per_cm = (w * 0.10) / self.scale_ref

        # Inferencia con SAM
        results = self.model(img)
        fragments = []
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                mask_uint8 = (mask * 255).astype(np.uint8)
                # Redimensionar máscara al tamaño original si SAM la devolvió reducida
                if mask_uint8.shape[:2] != (h, w):
                    mask_uint8 = cv2.resize(mask_uint8, (w, h))
                
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                
                cnt = max(contours, key=cv2.contourArea)
                area_px = cv2.contourArea(cnt)
                
                if self.min_px < area_px < (h * w * self.max_ratio):
                    area_cm2 = area_px / (scale_px_per_cm ** 2)
                    diameter = 2 * np.sqrt(area_cm2 / np.pi)
                    fragments.append(Fragment(id=i, area_px=area_px, diameter_cm=diameter, contour=cnt, bbox=cv2.boundingRect(cnt)))

        if not fragments:
            raise ValueError("No se detectaron fragmentos con los filtros actuales.")

        diameters = sorted([f.diameter_cm for f in fragments])
        return AnalysisResult(
            total_fragments=len(fragments),
            p20=np.percentile(diameters, 20),
            p50=np.percentile(diameters, 50),
            p80=np.percentile(diameters, 80),
            mean_diameter=np.mean(diameters),
            max_diameter=max(diameters),
            fragments=fragments,
            scale_px_per_cm=scale_px_per_cm
        )

class RockFragVisualizer:
    @staticmethod
    def draw_segmentation(img, result):
        out = img.copy()
        for f in result.fragments:
            cv2.drawContours(out, [f.contour], -1, (0, 212, 255), 2)
        return out

    @staticmethod
    def plot_grading_curve(result):
        diams = sorted([f.diameter_cm for f in result.fragments])
        y = [(i+1)/len(diams)*100 for i in range(len(diams))]
        fig, ax = plt.subplots()
        ax.plot(diams, y, color='#00d4ff', linewidth=2)
        ax.set_title("Curva Granulométrica")
        ax.set_xlabel("Diámetro (cm)"); ax.set_ylabel("% Pasante")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        return buf.getvalue()

    @staticmethod
    def result_to_dict(result):
        return {"total": result.total_fragments, "p50": result.p50, "p80": result.p80}

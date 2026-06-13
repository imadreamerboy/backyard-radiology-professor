from __future__ import annotations


XRAYDAR_LABELS = [
    "abnormal_non_clinically_important",
    "aortic_calcification",
    "apical_changes",
    "atelectasis",
    "axillary_abnormality",
    "bronchial_changes",
    "bulla",
    "cardiomegaly",
    "cavity",
    "clavicle_fracture",
    "consolidation",
    "cardiac_calcification",
    "dextrocardia",
    "dilated_bowel",
    "emphysema",
    "ground_glass_opacification",
    "hemidiaphragm_elevated",
    "hernia",
    "hyperexpanded_lungs",
    "interstitial_shadowing",
    "mediastinum_displaced",
    "mediastinum_widened",
    "object",
    "paraspinal_mass",
    "paratracheal_hilar_enlargement",
    "parenchymal_lesion",
    "pleural_abnormality",
    "pleural_effusion",
    "pneumomediastinum",
    "pneumoperitoneum",
    "pneumothorax",
    "rib_fracture",
    "rib_lesion",
    "scoliosis",
    "subcutaneous_emphysema",
    "tortuosity_aorta",
    "pulmonary_bloodflow_redistribution",
    "volume_loss",
]


LABEL_ALIASES = {
    "unfolded_aorta": "tortuosity_aorta",
    "pulmonary_bloodflow_redis.": "pulmonary_bloodflow_redistribution",
}


def normalize_xraydar_label(label: str) -> str:
    normalized = str(label).strip().replace(" ", "_").lower()
    return LABEL_ALIASES.get(normalized, normalized)


def display_xraydar_label(label: str) -> str:
    return normalize_xraydar_label(label).replace("_", " ")

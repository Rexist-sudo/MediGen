"""Curated local medication-interaction and allergy safety rules."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# The local ruleset is deliberately bounded and auditable. It supplements the
# generated plan and its findings always remain subject to professional review.
DDI_DATABASE = [
    {
        "drug_a": "warfarin",
        "drug_b": "aspirin",
        "severity": "major",
        "description": "华法林与阿司匹林联用可增加出血风险",
        "recommendation": "核对联用指征，并加强出血征象与 INR 监测",
    },
    {
        "drug_a": "metformin",
        "drug_b": "contrast_dye",
        "severity": "major",
        "description": "二甲双胍与含碘对比剂相关的乳酸性酸中毒风险增加",
        "recommendation": "结合肾功能与检查安排，由临床人员制定停药和复药时间",
    },
    {
        "drug_a": "ssri",
        "drug_b": "maoi",
        "severity": "contraindicated",
        "description": "联用可引发严重的血清素综合征",
        "recommendation": "该组合属于禁忌，需核对停药间隔并由临床人员调整方案",
    },
    {
        "drug_a": "ace_inhibitor",
        "drug_b": "potassium_supplement",
        "severity": "moderate",
        "description": "联用可增加高钾血症风险",
        "recommendation": "复核补钾指征并监测血钾与肾功能",
    },
    {
        "drug_a": "simvastatin",
        "drug_b": "amiodarone",
        "severity": "major",
        "description": "联用可增加肌病与横纹肌溶解风险",
        "recommendation": "复核辛伐他汀剂量并监测肌肉症状和肌酸激酶",
    },
    {
        "drug_a": "ciprofloxacin",
        "drug_b": "antacid",
        "severity": "moderate",
        "description": "抗酸药可降低环丙沙星吸收",
        "recommendation": "由临床人员安排错开用药时间",
    },
    {
        "drug_a": "methotrexate",
        "drug_b": "nsaid",
        "severity": "major",
        "description": "非甾体抗炎药可降低甲氨蝶呤清除并增加毒性",
        "recommendation": "复核联用必要性，并监测血细胞计数与肾功能",
    },
    {
        "drug_a": "digoxin",
        "drug_b": "amiodarone",
        "severity": "major",
        "description": "胺碘酮可升高地高辛浓度并增加中毒风险",
        "recommendation": "由临床人员复核剂量并监测地高辛浓度和心率",
    },
    {
        "drug_a": "lithium",
        "drug_b": "nsaid",
        "severity": "major",
        "description": "非甾体抗炎药可升高锂盐浓度",
        "recommendation": "复核联用方案并监测血锂浓度和肾功能",
    },
    {
        "drug_a": "clopidogrel",
        "drug_b": "omeprazole",
        "severity": "moderate",
        "description": "奥美拉唑可能通过 CYP2C19 降低氯吡格雷活性",
        "recommendation": "由临床人员评估抑酸治疗选择",
    },
]

# Drug class mappings for fuzzy matching
DRUG_CLASS_MAP = {
    "lisinopril": "ace_inhibitor",
    "enalapril": "ace_inhibitor",
    "ramipril": "ace_inhibitor",
    "fluoxetine": "ssri",
    "sertraline": "ssri",
    "paroxetine": "ssri",
    "escitalopram": "ssri",
    "ibuprofen": "nsaid",
    "naproxen": "nsaid",
    "diclofenac": "nsaid",
    "celecoxib": "nsaid",
    "phenelzine": "maoi",
    "tranylcypromine": "maoi",
}

DRUG_ALIASES = {
    "华法林": "warfarin",
    "阿司匹林": "aspirin",
    "二甲双胍": "metformin",
    "含碘对比剂": "contrast_dye",
    "碘对比剂": "contrast_dye",
    "补钾": "potassium_supplement",
    "钾补充剂": "potassium_supplement",
    "辛伐他汀": "simvastatin",
    "胺碘酮": "amiodarone",
    "环丙沙星": "ciprofloxacin",
    "抗酸药": "antacid",
    "甲氨蝶呤": "methotrexate",
    "地高辛": "digoxin",
    "锂盐": "lithium",
    "氯吡格雷": "clopidogrel",
    "奥美拉唑": "omeprazole",
    "阿莫西林": "amoxicillin",
    "氨苄西林": "ampicillin",
}


def _normalize_drug(name: str) -> list[str]:
    """Return possible drug/class identifiers for matching."""
    lower = name.casefold().strip()
    candidates = {lower}
    for alias, canonical in DRUG_ALIASES.items():
        if alias.casefold() in lower:
            candidates.add(canonical)
    known_identifiers = {
        *DRUG_CLASS_MAP,
        *(
            item
            for row in DDI_DATABASE
            for item in (row["drug_a"], row["drug_b"])
        ),
    }
    for known in known_identifiers:
        if known in lower:
            candidates.add(known)
    candidates.update(
        DRUG_CLASS_MAP[item]
        for item in tuple(candidates)
        if item in DRUG_CLASS_MAP
    )
    return sorted(candidates)


def check_interactions(new_drugs: list[str], current_drugs: list[str]) -> list[dict]:
    """
    Check for drug-drug interactions between new prescriptions and current meds.
    Returns list of interaction records.
    """
    interactions = []

    all_new = []
    for d in new_drugs:
        all_new.extend(_normalize_drug(d))

    all_current = []
    for d in current_drugs:
        all_current.extend(_normalize_drug(d))

    for ddi in DDI_DATABASE:
        a, b = ddi["drug_a"], ddi["drug_b"]
        if (a in all_new and b in all_current) or (b in all_new and a in all_current):
            interactions.append(ddi)
        elif a in all_new and b in all_new:
            interactions.append(ddi)

    if interactions:
        logger.warning("ddi.found", count=len(interactions))
    return interactions


def check_allergy_contraindication(drug: str, allergies: list[str]) -> dict | None:
    """Check if a drug conflicts with known allergies."""
    normalized_drugs = set(_normalize_drug(drug))
    for allergy in allergies:
        allergy_lower = allergy.casefold().strip()
        drug_lower = drug.casefold().strip()
        if drug_lower in allergy_lower or allergy_lower in drug_lower:
            return {
                "drug": drug,
                "allergy": allergy,
                "severity": "contraindicated",
                "recommendation": f"{drug} 与既往 {allergy} 过敏史冲突，需停止给药并由临床人员复核",
            }
        penicillin_allergy = "penicillin" in allergy_lower or "青霉素" in allergy_lower
        if penicillin_allergy and normalized_drugs.intersection({"amoxicillin", "ampicillin"}):
            return {
                "drug": drug,
                "allergy": allergy,
                "severity": "major",
                "recommendation": f"{drug} 与青霉素过敏史存在交叉过敏风险，需停止给药并由临床人员复核",
            }
    return None

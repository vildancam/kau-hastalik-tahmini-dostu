from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Referral:
    branch: str
    message: str
    urgent: bool = False


URGENT_PATTERNS = (
    r"acil",
    r"kalp krizi",
    r"inme",
    r"felc",
    r"felç",
    r"apandisit",
    r"zehirlen",
    r"anafilaksi",
    r"kanama",
)

BRANCH_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("migren", "bas agrisi", "baş ağrısı", "epilepsi", "sinir"), "Nöroloji"),
    (("mide", "gastrit", "reflu", "reflü", "bagirsak", "bağırsak", "ishal", "ulser", "ülser"), "Gastroenteroloji"),
    (("grip", "nezle", "soguk alginligi", "soğuk algınlığı", "ates", "ateş", "enfeksiyon"), "Aile Hekimi / Dahiliye"),
    (("astim", "astım", "bronşit", "bronsit", "zatürre", "zaturre", "akciger", "akciğer"), "Göğüs Hastalıkları"),
    (("alerji", "alerjik", "kurdesen", "ürtiker"), "Alerji ve İmmünoloji"),
    (("cilt", "egzama", "sedef", "deri", "akne"), "Dermatoloji"),
    (("idrar", "bobrek", "böbrek", "sistit", "prostat"), "Üroloji"),
    (("diyabet", "seker", "şeker", "tiroid", "hormon"), "Endokrinoloji"),
    (("romatizma", "eklem", "kas", "bel", "fitik", "fıtık"), "Fizik Tedavi ve Rehabilitasyon"),
    (("depresyon", "anksiyete", "panik", "uykusuzluk"), "Psikiyatri"),
    (("kulak", "burun", "bogaz", "boğaz", "sinuzit", "sinüzit"), "Kulak Burun Boğaz"),
    (("goz", "göz", "konjonktivit"), "Göz Hastalıkları"),
)


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = value.translate(str.maketrans("ıİğüşıöçĞÜŞÖÇ", "iigusiocGUSOC"))
    return re.sub(r"[^a-z0-9]+", " ", value)


def classify_referral(disease: str) -> Referral:
    original = disease.strip() or "tahmin edilen durum"
    normalized = normalize(original)

    if any(re.search(pattern, normalized) for pattern in URGENT_PATTERNS):
        return Referral(
            branch="ACİL SERVİS",
            message="ACİL SERVİS’e başvurmanız önerilir.",
            urgent=True,
        )

    branch = "Aile Hekimi / Dahiliye"
    disease_key = f" {normalized} "
    for keywords, candidate in BRANCH_PATTERNS:
        if any(normalize(keyword).strip() in disease_key for keyword in keywords):
            branch = candidate
            break

    return Referral(
        branch=branch,
        message=f"Tahmin edilen hastalık nedeniyle {branch} Uzmanı ile görüşmeniz önerilir.",
    )


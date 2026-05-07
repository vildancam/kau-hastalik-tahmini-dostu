from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

import requests
from bs4 import BeautifulSoup

from app.config import PHARMACY_URL

LOGGER = logging.getLogger(__name__)
PHONE_RE = re.compile(r"0\s?\(\d{3}\)\s?\d{3}[-\s]?\d{2}[-\s]?\d{2}")

_CACHE: dict[str, Any] = {"expires_at": None, "items": []}


def fetch_kars_pharmacies(ttl_minutes: int = 30) -> list[dict[str, str]]:
    now = datetime.utcnow()
    if _CACHE["expires_at"] and _CACHE["expires_at"] > now:
        return _CACHE["items"]

    response = requests.get(
        PHARMACY_URL,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 KAU-Hastalik-Tahmini-Dostu/1.0"},
    )
    response.raise_for_status()
    items = parse_pharmacies(response.text)
    _CACHE["items"] = items
    _CACHE["expires_at"] = now + timedelta(minutes=ttl_minutes)
    return items


def parse_pharmacies(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select(".eczane, .pharmacy, .nobetci, .card")
    items = _parse_structured_rows(rows)
    if items:
        return items
    return _parse_text_fallback(soup.get_text("\n", strip=True))


def _parse_structured_rows(rows) -> list[dict[str, str]]:
    pharmacies: list[dict[str, str]] = []
    for row in rows:
        text = row.get_text("\n", strip=True)
        phone_match = PHONE_RE.search(text)
        if not phone_match:
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        name = lines[0].replace("Eczanesi Eczanesi", "Eczanesi")
        phone = phone_match.group(0)
        address_lines = [line for line in lines[1:] if phone not in line]
        district = address_lines[-1] if address_lines else "Kars"
        pharmacies.append(
            {
                "name": name,
                "address": " ".join(address_lines[:-1] or address_lines),
                "district": district,
                "phone": phone,
                "source": PHARMACY_URL,
            }
        )
    return _dedupe(pharmacies)


def _parse_text_fallback(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = _current_duty_section(lines)
    pharmacies: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        if not line.endswith("Eczanesi"):
            continue
        window = []
        phone = ""
        for item in lines[index + 1 : index + 10]:
            phone_match = PHONE_RE.search(item)
            if phone_match:
                phone = phone_match.group(0)
                break
            if item.endswith("Eczanesi"):
                break
            window.append(item)
        if not phone:
            continue
        details = [item.lstrip("» ").strip() for item in window if not item.startswith("Image:")]
        district = details[-1] if details else "Kars"
        address = " ".join(details[:-1] or details)
        pharmacies.append(
            {
                "name": line,
                "address": address,
                "district": district,
                "phone": phone,
                "source": PHARMACY_URL,
            }
        )
    return _dedupe(pharmacies)


def _current_duty_section(lines: list[str]) -> list[str]:
    section_indexes = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\d{1,2}\s+\S+\s+.*akşamından", line, flags=re.IGNORECASE)
    ]
    if not section_indexes:
        return lines

    today = datetime.now().day
    selected = section_indexes[0]
    for index in section_indexes:
        if lines[index].startswith(f"{today} "):
            selected = index
            break

    next_indexes = [index for index in section_indexes if index > selected]
    end = next_indexes[0] if next_indexes else len(lines)
    return lines[selected:end]


def _dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    unique = []
    for item in items:
        key = (item["name"], item["phone"], item["district"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique

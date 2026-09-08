"""
Verify every citation before it enters the bibliography, and again after the .bib is written.
ML4PS 2026 does not send papers with erroneous citations out for review, so each entry is resolved
against the arXiv API; the returned title, author count and date are written to refs_verified.txt
so the .bib can be diffed against ground truth rather than trusted.
"""

from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

atom = "{http://www.w3.org/2005/Atom}"

# Final citation list: bibtex key --> arXiv identifier.
cited_ids = {
    "ricker2015": "1406.0151",
    "audenaert2021": "2107.06301",
    "prsa2022": "2110.13382",
    "gao2025": "2412.06175",
    "hon2021": "2108.01241",
    "hatt2023": "2210.09109",
    "boyle2026": "2603.05586",
    "seli2025": "2412.12989",
    "vida2021": "2105.11485",
    "zuo2025falco": "2504.20290",
    "donoso2025astromer2": "2502.02717",
    "li2025staremb": "2510.06200",
    "rui2026jepa": "2606.28446",
    "poznanski2026": "2605.05324",
    "makinen2024": "2410.07548",
    "ivezic2019": "0805.2366",
    "kingma2014": "1312.6114",
    "cho2014": "1406.1078",
    # Pinned 2026-09-06. Both were located by the field-scoped searches below and their ids then
    # entered refs.bib, but they were never moved out of `open_queries`, so the report showed 18 [OK]
    # rows and a trailing "unresolved" block that made two settled entries look outstanding.
    "sreenivas2026": "2604.00498",
    "guerrero2021": "2103.12538",
    # Added 2026-09-07 with Appendix A: the physical discriminant behind the rgb_vs_heb labels.
    "bedding2011": "1103.5805",
}

# Kept so the report records HOW the two ids above were found, not because anything is outstanding.
open_queries = {
    "sreenivas_rgb_heb": 'au:Sreenivas AND abs:"red giant"',
    "guerrero_toi": 'au:Guerrero AND ti:"Objects of Interest"',
}


def fetch(query: str) -> list[dict[str, str]]:
    """
    Run one arXiv API query and return the parsed entries.
    Retries on transient network failure; arXiv asks for no more than one request every 3 seconds.
    """
    url = "http://export.arxiv.org/api/query?" + query
    payload = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            break
        except Exception as error:
            print(f"    retry {attempt} after {type(error).__name__}")
            time.sleep(8)
    if payload is None:
        return []
    root = ET.fromstring(payload)
    entries = []
    for entry in root.findall(f"{atom}entry"):
        authors = []
        for author in entry.findall(f"{atom}author"):
            authors.append(author.find(f"{atom}name").text)
        entries.append(
            {
                "id": entry.find(f"{atom}id").text.rsplit("/", 1)[-1],
                "title": " ".join(entry.find(f"{atom}title").text.split()),
                "authors": authors,
                "published": entry.find(f"{atom}published").text[:10],
            }
        )
    return entries


report = []
for key, arxiv_id in cited_ids.items():
    entries = fetch("id_list=" + arxiv_id)
    if not entries:
        report.append(f"[MISSING] {key}: {arxiv_id} did not resolve")
    else:
        found = entries[0]
        shown = ", ".join(found["authors"][:3])
        if len(found["authors"]) > 3:
            shown = shown + f", +{len(found['authors']) - 3} more"
        report.append(f"[OK] {key} | arXiv:{arxiv_id} | {found['published']}")
        report.append(f"     title:   {found['title']}")
        report.append(f"     authors: {shown}")
    time.sleep(3.5)

report.append("")
report.append("=== provenance of the two ids found by search, not by id lookup ===")
for key, query in open_queries.items():
    entries = fetch(f"search_query={urllib.parse.quote(query)}&max_results=4")
    report.append(f"--- {key}")
    for found in entries:
        report.append(f"    {found['id']} | {found['authors'][0]} | {found['published']} | {found['title']}")
    time.sleep(3.5)

text = "\n".join(report)
# Write before printing: author names carry accents (Ivezic, Prsa, Seli) and a Windows console on
# cp1252 raises on them, which previously killed the run after every query and left the file stale.
Path("paper/build/refs_verified.txt").write_text(text + "\n", encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(text)

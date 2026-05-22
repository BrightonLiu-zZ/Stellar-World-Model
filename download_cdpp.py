"""
download_cdpp.py
================
Downloads TESS SPOC 2-min CDPP CSV files from MAST and builds a
sector-availability lookup table for your TIC list.

Output files (all saved in the same directory as this script):
  cdpp_raw/          — raw per-sector CSV files from MAST
  spoc_sector_map.csv — final lookup: tic_id | sector | tmag

Usage:
  python download_cdpp.py                  # full run (sectors 1-101)
  python download_cdpp.py --tmag-max 10    # only keep Tmag < 10 (default)
  python download_cdpp.py --sectors 1-26   # only download sectors 1 to 26
  python download_cdpp.py --resume         # skip already-downloaded sectors
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Complete filename list extracted from MAST TCE Bulk Downloads page
# (https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html)
# Base URL for all CDPP CSVs:
BASE_URL = "https://archive.stsci.edu/missions/tess/catalogs/cdpp/"

# Sector -> filename mapping (single-sector, sectors 1-101)
SECTOR_FILES = {
    1:   "tess2018206190146-s0001-s0001-00366_rms-cdpp.csv",
    2:   "tess2018235142537-s0002-s0002-00372_rms-cdpp.csv",
    3:   "tess2018263124742-s0003-s0003-00405_rms-cdpp.csv",
    4:   "tess2018292093535-s0004-s0004-00407_rms-cdpp.csv",
    5:   "tess2018319112537-s0005-s0005-00413_rms-cdpp.csv",
    6:   "tess2018349182740-s0006-s0006-00415_rms-cdpp.csv",
    7:   "tess2019008025933-s0007-s0007-00396_rms-cdpp.csv",
    8:   "tess2019033200937-s0008-s0008-00401_rms-cdpp.csv",
    9:   "tess2019059170937-s0009-s0009-00403_rms-cdpp.csv",
    10:  "tess2019085221931-s0010-s0010-00446_rms-cdpp.csv",
    11:  "tess2019113062936-s0011-s0011-00442_rms-cdpp.csv",
    12:  "tess2019141104532-s0012-s0012-00437_rms-cdpp.csv",
    13:  "tess2019170095526-s0013-s0013-00439_rms-cdpp.csv",
    14:  "tess2019199201928-s0014-s0014-00745_rms-cdpp.csv",
    15:  "tess2019227203528-s0015-s0015-00752_rms-cdpp.csv",
    16:  "tess2019255032925-s0016-s0016-00754_rms-cdpp.csv",
    17:  "tess2019281041529-s0017-s0017-00758_rms-cdpp.csv",
    18:  "tess2019307033524-s0018-s0018-00759_rms-cdpp.csv",
    19:  "tess2019332134924-s0019-s0019-00760_rms-cdpp.csv",
    20:  "tess2019358235519-s0020-s0020-00761_rms-cdpp.csv",
    21:  "tess2020021221518-s0021-s0021-00763_rms-cdpp.csv",
    22:  "tess2020050191126-s0022-s0022-00765_rms-cdpp.csv",
    23:  "tess2020079142124-s0023-s0023-00766_rms-cdpp.csv",
    24:  "tess2020107065516-s0024-s0024-00769_rms-cdpp.csv",
    25:  "tess2020135030118-s0025-s0025-00771_rms-cdpp.csv",
    26:  "tess2020161181522-s0026-s0026-00773_rms-cdpp.csv",
    27:  "tess2020187183111-s0027-s0027-00360_rms-cdpp.csv",
    28:  "tess2020213081515-s0028-s0028-00364_rms-cdpp.csv",
    29:  "tess2020239173517-s0029-s0029-00382_rms-cdpp.csv",
    30:  "tess2020267090510-s0030-s0030-00393_rms-cdpp.csv",
    31:  "tess2020296001109-s0031-s0031-00411_rms-cdpp.csv",
    32:  "tess2020325171308-s0032-s0032-00419_rms-cdpp.csv",
    33:  "tess2020353052505-s0033-s0033-00430_rms-cdpp.csv",
    34:  "tess2021014055106-s0034-s0034-00444_rms-cdpp.csv",
    35:  "tess2021040113517-s0035-s0035-00453_rms-cdpp.csv",
    36:  "tess2021066093111-s0036-s0036-00459_rms-cdpp.csv",
    37:  "tess2021092173456-s0037-s0037-00478_rms-cdpp.csv",
    38:  "tess2021119082113-s0038-s0038-00488_rms-cdpp.csv",
    39:  "tess2021147062059-s0039-s0039-00491_rms-cdpp.csv",
    40:  "tess2021176033111-s0040-s0040-00503_rms-cdpp.csv",
    41:  "tess2021205113456-s0041-s0041-00511_rms-cdpp.csv",
    42:  "tess2021233042458-s0042-s0042-00516_rms-cdpp.csv",
    43:  "tess2021259155052-s0043-s0043-00521_rms-cdpp.csv",
    44:  "tess2021285162106-s0044-s0044-00532_rms-cdpp.csv",
    45:  "tess2021311000057-s0045-s0045-00542_rms-cdpp.csv",
    46:  "tess2021337012458-s0046-s0046-00546_rms-cdpp.csv",
    47:  "tess2021365070456-s0047-s0047-00559_rms-cdpp.csv",
    48:  "tess2022028101447-s0048-s0048-00779_rms-cdpp.csv",
    49:  "tess2022057231102-s0049-s0049-00781_rms-cdpp.csv",
    50:  "tess2022085182059-s0050-s0050-00782_rms-cdpp.csv",
    51:  "tess2022113103449-s0051-s0051-00783_rms-cdpp.csv",
    52:  "tess2022139030449-s0052-s0052-00784_rms-cdpp.csv",
    53:  "tess2022164114447-s0053-s0053-00786_rms-cdpp.csv",
    54:  "tess2022190092451-s0054-s0054-00799_rms-cdpp.csv",
    55:  "tess2022217141454-s0055-s0055-00788_rms-cdpp.csv",
    56:  "tess2022245180036-s0056-s0056-00789_rms-cdpp.csv",
    57:  "tess2022273202052-s0057-s0057-00790_rms-cdpp.csv",
    58:  "tess2022302194440-s0058-s0058-00791_rms-cdpp.csv",
    59:  "tess2022330181048-s0059-s0059-00793_rms-cdpp.csv",
    60:  "tess2022357093050-s0060-s0060-00794_rms-cdpp.csv",
    61:  "tess2023018070043-s0061-s0061-00795_rms-cdpp.csv",
    62:  "tess2023043222437-s0062-s0062-00796_rms-cdpp.csv",
    63:  "tess2023069204033-s0063-s0063-00797_rms-cdpp.csv",
    64:  "tess2023096143435-s0064-s0064-00720_rms-cdpp.csv",
    65:  "tess2023124053435-s0065-s0065-00798_rms-cdpp.csv",
    66:  "tess2023153040043-s0066-s0066-00749_rms-cdpp.csv",
    67:  "tess2023182031440-s0067-s0067-00756_rms-cdpp.csv",
    68:  "tess2023210023043-s0068-s0068-00777_rms-cdpp.csv",
    69:  "tess2023237202031-s0069-s0069-00803_rms-cdpp.csv",
    70:  "tess2023263202031-s0070-s0070-00809_rms-cdpp.csv",
    71:  "tess2023289124019-s0071-s0071-00816_rms-cdpp.csv",
    72:  "tess2023315161425-s0072-s0072-00827_rms-cdpp.csv",
    73:  "tess2023341070022-s0073-s0073-00835_rms-cdpp.csv",
    74:  "tess2024003083435-s0074-s0074-00841_rms-cdpp.csv",
    75:  "tess2024030064019-s0075-s0075-00848_rms-cdpp.csv",
    76:  "tess2024058043428-s0076-s0076-00856_rms-cdpp.csv",
    77:  "tess2024085233022-s0077-s0077-00867_rms-cdpp.csv",
    78:  "tess2024124181749-s0078-s0078-00875_rms-cdpp.csv",
    79:  "tess2024143004536-s0079-s0079-00890_rms-cdpp.csv",
    80:  "tess2024170090944-s0080-s0080-00899_rms-cdpp.csv",
    81:  "tess2024197010538-s0081-s0081-00910_rms-cdpp.csv",
    82:  "tess2024223211543-s0082-s0082-00919_rms-cdpp.csv",
    83:  "tess2024249220538-s0083-s0083-00927_rms-cdpp.csv",
    84:  "tess2024275015925-s0084-s0084-00942_rms-cdpp.csv",
    85:  "tess2024301010930-s0085-s0085-00946_rms-cdpp.csv",
    86:  "tess2024326180531-s0086-s0086-00952_rms-cdpp.csv",
    87:  "tess2024353124928-s0087-s0087-00961_rms-cdpp.csv",
    88:  "tess2025014152925-s0088-s0088-00969_rms-cdpp.csv",
    89:  "tess2025042150923-s0089-s0089-00976_rms-cdpp.csv",
    90:  "tess2025071153929-s0090-s0090-00989_rms-cdpp.csv",
    91:  "tess2025099184928-s0091-s0091-01000_rms-cdpp.csv",
    92:  "tess2025127110927-s0092-s0092-01014_rms-cdpp.csv",
    93:  "tess2025154082526-s0093-s0093-01019_rms-cdpp.csv",
    94:  "tess2025180180925-s0094-s0094-01025_rms-cdpp.csv",
    95:  "tess2025206194924-s0095-s0095-01037_rms-cdpp.csv",
    96:  "tess2025232062523-s0096-s0096-01048_rms-cdpp.csv",
    97:  "tess2025258033922-s0097-s0097-01067_rms-cdpp.csv",
    98:  "tess2025312234920-s0098-s0098-01083_rms-cdpp.csv",
    99:  "tess2026005143517-s0099-s0099-01089_rms-cdpp.csv",
    100: "tess2026033114116-s0100-s0100-01101_rms-cdpp.csv",
    101: "tess2026060041115-s0101-s0101-01111_rms-cdpp.csv",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tmag-max", type=float, default=10.0,
                   help="Keep only stars with tmag <= this value (default: 10.0)")
    p.add_argument("--sectors", default=None,
                   help="Sector range to download, e.g. '1-26' or '1-101' (default: all)")
    p.add_argument("--resume", action="store_true",
                   help="Skip sectors whose CSV is already downloaded")
    p.add_argument("--out-dir", default=".", help="Directory to save output files")
    return p.parse_args()


def download_sector(sector: int, filename: str, raw_dir: Path, resume: bool) -> Path | None:
    dest = raw_dir / filename
    if resume and dest.exists():
        print(f"  Sector {sector:3d}: already downloaded, skipping")
        return dest

    url = BASE_URL + filename
    print(f"  Sector {sector:3d}: downloading {filename} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        kb = len(r.content) / 1024
        print(f"OK ({kb:.0f} KB)")
        return dest
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def load_cdpp_csv(path: Path, sector: int) -> pd.DataFrame:
    """Read a CDPP CSV, keep only ticid + tmag, add sector column."""
    df = pd.read_csv(path, comment="#", skipinitialspace=True, usecols=["ticid", "tmag"])
    df = df.rename(columns={"ticid": "tic_id"})
    df["sector"] = sector
    return df


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "data" / "cdpp_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Parse sector range
    if args.sectors:
        lo, hi = args.sectors.split("-")
        sectors_to_do = {s: f for s, f in SECTOR_FILES.items()
                         if int(lo) <= s <= int(hi)}
    else:
        sectors_to_do = SECTOR_FILES

    print(f"Downloading {len(sectors_to_do)} sector CDPP files to {raw_dir}/")
    print(f"Tmag filter: <= {args.tmag_max}")
    print()

    chunks = []
    n_ok = 0
    n_fail = 0

    for sector, filename in sorted(sectors_to_do.items()):
        path = download_sector(sector, filename, raw_dir, args.resume)
        if path is None:
            n_fail += 1
            continue
        try:
            df = load_cdpp_csv(path, sector)
            df = df[df["tmag"] <= args.tmag_max]
            chunks.append(df)
            n_ok += 1
        except Exception as e:
            print(f"  Sector {sector:3d}: parse error: {e}")
            n_fail += 1
        time.sleep(0.1)  # polite rate limiting

    if not chunks:
        print("No data loaded. Check network or sector range.")
        return

    print()
    print("Building sector availability map ...")
    master = pd.concat(chunks, ignore_index=True)
    master = master.sort_values(["tic_id", "sector"]).reset_index(drop=True)

    out_path = out_dir / "spoc_sector_map.csv"
    master.to_csv(out_path, index=False)

    n_stars = master["tic_id"].nunique()
    print(f"Done.")
    print(f"  Sectors downloaded OK : {n_ok}")
    print(f"  Sectors failed        : {n_fail}")
    print(f"  Unique TICs (Tmag<={args.tmag_max}): {n_stars:,}")
    print(f"  Total (tic_id, sector) rows: {len(master):,}")
    print(f"  Output saved to: {out_path}")


if __name__ == "__main__":
    main()

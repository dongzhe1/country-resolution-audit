#!/usr/bin/env python3
"""Synthesise a small dataset with the shape of the real inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

VISIT_FIELDS = [
    "imo", "event_id", "start", "end", "duration_hrs", "confidence",
    "lat", "lon",
    "vessel_id", "ssvid", "vessel_name", "vessel_flag", "vessel_type",
    "start_anchorage_id", "start_anchorage_name", "start_anchorage_flag",
    "start_at_dock", "start_top_destination",
    "end_anchorage_id", "end_anchorage_name", "end_anchorage_flag",
    "end_at_dock", "end_top_destination",
]

PORTS = [
    ("CHN", "SHANGHAI", 31.2, 121.5), ("USA", "LOS ANGELES", 33.7, -118.3),
    ("NLD", "ROTTERDAM", 51.9, 4.1), ("SGP", "SINGAPORE", 1.26, 103.8),
    ("MYS", "PASIR GUDANG", 1.44, 103.9), ("BRA", "SANTOS", -23.9, -46.3),
    ("ZAF", "DURBAN", -29.9, 31.0), ("IND", "MUMBAI", 18.9, 72.8),
    ("AUS", "MELBOURNE", -37.8, 144.9), ("ESP", "ALGECIRAS", 36.1, -5.4),
    ("GIB", "GIBRALTAR", 36.1, -5.35), ("PAN", "BALBOA", 8.9, -79.6),
    ("EGY", "SUEZ", 29.9, 32.5), ("TGO", "LOME", 6.1, 1.3),
    ("DJI", "DJIBOUTI", 11.6, 43.1), ("JAM", "KINGSTON", 18.0, -76.8),
    ("FJI", "SUVA", -18.1, 178.4), ("MDG", "TOAMASINA", -18.1, 49.4),
    ("SEN", "DAKAR", 14.7, -17.4), ("VUT", "PORT VILA", -17.7, 168.3),
]

TYPES = [
    ("Bulk Carrier-Supra/Ultramax", 61000, 32000, 9500, 14.5),
    ("Containership-Post-Panamax", 100000, 90000, 55000, 24.5),
    ("Tanker-MR2", 50000, 30000, 9000, 14.5),
    ("Tanker-VLCC", 300000, 160000, 27000, 13.0),
    ("General Cargo 5,000-9,999 dwt", 8000, 5000, 3000, 13.3),
    ("LNG-New Panamax", 90000, 110000, 30000, 19.5),
]


def haversine_nm(a, b, c, d):
    r = 3440.065
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - b)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--vessels", type=int, default=400)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "mrv").mkdir(exist_ok=True)

    imos = [f"9{600000 + i}" for i in range(a.vessels)]
    fleet = {imo: TYPES[rng.randrange(len(TYPES))] for imo in imos}
    flags = [p[0] for p in PORTS]

    with open(a.out / "ship_info.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lrnoimo_ship_no", "gross_tonnage", "deadweight",
                    "shiptype_group", "total_kilowattsof_main_engines",
                    "speedservice", "year_of_build", "flag_name"])
        for imo in imos:
            grp, dwt, gt, kw, svc = fleet[imo]
            j = rng.uniform(0.85, 1.15)
            w.writerow([imo, round(gt * j), round(dwt * j), grp,
                        round(kw * j), svc, rng.randint(1998, 2022),
                        flags[rng.randrange(len(flags))]])

    shards = [gzip.open(a.out / f"port_visits_{i:04d}.csv.gz", "wt", newline="")
              for i in range(a.shards)]
    writers = []
    for fh in shards:
        w = csv.DictWriter(fh, fieldnames=VISIT_FIELDS)
        w.writeheader()
        writers.append(w)

    t0 = datetime(2018, 1, 1)
    span_days = (datetime(2025, 6, 1) - t0).days
    n_events = 0
    itineraries = {}
    for k, imo in enumerate(imos):
        w = writers[k % a.shards]
        t = t0 + timedelta(days=rng.randrange(0, span_days))
        here = rng.randrange(len(PORTS))
        seen = set()
        for _ in range(rng.randint(14, 40)):
            iso, name, lat, lon = PORTS[here]
            seen.add(iso)
            stay = rng.uniform(12, 96)
            end = t + timedelta(hours=stay)
            if rng.random() < 0.03:
                other = rng.randrange(len(PORTS))
                s_iso, s_name = PORTS[other][0], PORTS[other][1]
            else:
                s_iso, s_name = iso, name
            w.writerow({
                "imo": imo, "event_id": f"e{n_events}",
                "start": t.isoformat() + "Z", "end": end.isoformat() + "Z",
                "duration_hrs": f"{stay:.2f}", "confidence": "4",
                "lat": f"{lat:.4f}", "lon": f"{lon:.4f}",
                "vessel_id": f"v{k}", "ssvid": f"2{k:08d}",
                "vessel_name": f"SHIP {k}",
                "vessel_flag": flags[k % len(flags)],
                "vessel_type": "CARGO",
                "start_anchorage_id": f"{s_iso}-{s_name}",
                "start_anchorage_name": s_name, "start_anchorage_flag": s_iso,
                "start_at_dock": "true", "start_top_destination": name,
                "end_anchorage_id": f"{iso}-{name}",
                "end_anchorage_name": name, "end_anchorage_flag": iso,
                "end_at_dock": "true", "end_top_destination": name,
            })
            n_events += 1
            nxt = rng.randrange(len(PORTS))
            d = haversine_nm(lat, lon, PORTS[nxt][2], PORTS[nxt][3])
            t = end + timedelta(hours=max(6.0, d / rng.uniform(9, 15)))
            here = nxt
        itineraries[imo] = seen
    for fh in shards:
        fh.close()

    eea_ports = {"NLD", "ESP"}
    eea = sorted({imo for imo, ports in itineraries.items() if ports & eea_ports})
    for year in (2022, 2023, 2024):
        with open(a.out / "mrv" / f"{year}_renamed.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["imo_number", "total_co2_tonnes", "year"])
            for imo in eea:
                grp, dwt, gt, kw, svc = fleet[imo]
                base = kw * 0.55 * 190e-6 * 3.114 * 3500
                w.writerow([imo, f"{base * rng.uniform(0.7, 1.4):.1f}", year])

    print(f"wrote {n_events:,} port-call events for {len(imos)} vessels "
          f"in {a.shards} shard(s)")
    print(f"  {a.out}/port_visits_*.csv.gz")
    print(f"  {a.out}/ship_info.csv          {len(imos)} vessels")
    print(f"  {a.out}/mrv/*.csv              {len(eea)} ships x 3 years")
    print("\nThese values are random. They run the code; they reproduce nothing.")


if __name__ == "__main__":
    main()

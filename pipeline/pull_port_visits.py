#!/usr/bin/env python3
"""Pull global AIS port-visit events from Global Fishing Watch, vessel by vessel.

    python pull_port_visits.py /path/to/work_dir
"""

from __future__ import annotations

import collections
import csv
import gzip
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://gateway.api.globalfishingwatch.org/v3"
IDENTITY_DS = "public-global-vessel-identity:latest"
PORTVISIT_DS = "public-global-port-visits-events:latest"
START_DATE = "2018-01-01"
END_DATE = "2026-08-01"
PAGE = 500
SHARD_ROWS = 500_000
MAX_RETRIES = 8
MAX_429_SLEEPS = 8
BASE_BACKOFF = 2.0
PAUSE = 0.12
USER_AGENT = "port-visit-extraction/1.0 (academic research; GFW API v3)"

MIN_CONFIDENCE = 3
MAX_DURATION_HRS = 24 * 30
KEEP_ALL_ROWS = False

FIELDS = [
    "imo", "event_id", "start", "end", "duration_hrs", "confidence",
    "lat", "lon",
    "vessel_id", "ssvid", "vessel_name", "vessel_flag", "vessel_type",
    "start_anchorage_id", "start_anchorage_name", "start_anchorage_flag",
    "start_at_dock", "start_top_destination",
    "end_anchorage_id", "end_anchorage_name", "end_anchorage_flag",
    "end_at_dock", "end_top_destination",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_consec_429 = [0]


def api_get(token: str, path: str, params: list[tuple[str, str]]) -> dict:
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params, safe=":")
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token, "User-Agent": USER_AGENT}
    )
    delay = BASE_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            _consec_429[0] = 0
            return d
        except (urllib.error.HTTPError, http.client.HTTPException,
                OSError, json.JSONDecodeError) as exc:
            code = getattr(exc, "code", None)
            if code in (401, 403) and attempt >= 2:
                raise SystemExit(f"auth/permission failure ({code}) -- check GFW_TOKEN")
            if code == 422:
                try:
                    detail = json.loads(exc.read().decode())
                except Exception:
                    detail = {}
                raise SystemExit(f"422 from {path}: "
                                 f"{detail.get('messages') or detail}\n  {url}")
            if code == 429:
                _consec_429[0] += 1
                if _consec_429[0] == 1:
                    try:
                        log(f"  429 body: {exc.read().decode()[:400]}")
                    except Exception:
                        pass
                if _consec_429[0] > MAX_429_SLEEPS:
                    raise SystemExit(
                        f"429 for {MAX_429_SLEEPS * 15} min straight -- this is a "
                        f"disabled token or a monthly cap, not a daily one.\n"
                        f"  Progress is checkpointed; rerun when it is restored.")
                wait = 900
                try:
                    ra = int(exc.headers.get("Retry-After", ""))
                    wait = max(1, min(ra, 3600))
                except (TypeError, ValueError, AttributeError):
                    pass
                log(f"  429 ({_consec_429[0]}/{MAX_429_SLEEPS}) -- sleeping {wait}s")
                time.sleep(wait)
                continue
            if attempt == MAX_RETRIES:
                log(f"  giving up on {path} after {MAX_RETRIES} tries: {exc}")
                return None
            log(f"  retry {attempt}/{MAX_RETRIES} ({type(exc).__name__} "
                f"{code or ''}) -- sleeping {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)
    return None


def resolve_ids(token: str, imos: list[str], cache: Path) -> dict[str, str]:
    known: dict[str, str] = {}
    seen: set[str] = set()
    if cache.exists():
        with open(cache, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seen.add(row["imo"])
                if row["vessel_id"]:
                    known[row["imo"]] = row["vessel_id"]
        log(f"id cache: {len(seen):,} looked up, {len(known):,} matched")

    todo = [i for i in imos if i not in seen]
    if not todo:
        return known

    log(f"resolving {len(todo):,} IMOs -> vesselId")
    new = not cache.exists()
    with open(cache, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["imo", "vessel_id", "ssvid", "shipname", "flag"])
        for n, imo in enumerate(todo, 1):
            d = api_get(token, "vessels/search", [
                ("query", imo), ("datasets[0]", IDENTITY_DS), ("limit", "1"),
            ])
            if d is None:
                log(f"  unresolved (will retry): IMO {imo}")
                continue
            entries = d.get("entries") or []
            si = (entries[0].get("selfReportedInfo") or [{}])[0] if entries else {}
            vid = si.get("id") or ""
            if vid:
                known[imo] = vid
                w.writerow([imo, vid, si.get("ssvid", ""),
                            si.get("shipname", ""), si.get("flag", "")])
            else:
                w.writerow([imo, "", "", "", ""])
            fh.flush()
            if n % 200 == 0:
                log(f"  resolved {n:,}/{len(todo):,}")
            time.sleep(PAUSE)
    log(f"resolved {len(known):,} of {len(imos):,} IMOs")
    return known


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flatten(e: dict, imo: str) -> dict:
    pv = e.get("port_visit") or {}
    v = e.get("vessel") or {}
    pos = e.get("position") or {}
    sa = pv.get("startAnchorage") or {}
    ea = pv.get("endAnchorage") or {}
    return {
        "imo": imo,
        "event_id": e.get("id"), "start": e.get("start"), "end": e.get("end"),
        "duration_hrs": pv.get("durationHrs"), "confidence": pv.get("confidence"),
        "lat": pos.get("lat"), "lon": pos.get("lon"),
        "vessel_id": v.get("id"), "ssvid": v.get("ssvid"),
        "vessel_name": v.get("name"), "vessel_flag": v.get("flag"),
        "vessel_type": v.get("type"),
        "start_anchorage_id": sa.get("id"), "start_anchorage_name": sa.get("name"),
        "start_anchorage_flag": sa.get("flag"), "start_at_dock": sa.get("atDock"),
        "start_top_destination": sa.get("topDestination"),
        "end_anchorage_id": ea.get("id"), "end_anchorage_name": ea.get("name"),
        "end_anchorage_flag": ea.get("flag"), "end_at_dock": ea.get("atDock"),
        "end_top_destination": ea.get("topDestination"),
    }


def keep(row: dict) -> tuple[bool, str]:
    if KEEP_ALL_ROWS:
        return True, ""
    c = _num(row.get("confidence"))
    if c is not None and c < MIN_CONFIDENCE:
        return False, "low_confidence"
    d = _num(row.get("duration_hrs"))
    if d is not None and d > MAX_DURATION_HRS:
        return False, "implausible_duration"
    return True, ""


class ShardWriter:
    def __init__(self, outdir: Path, index: int = 0):
        self.outdir, self.index, self.rows = outdir, index, 0
        self.fh = self.writer = None

    def _open(self):
        path = self.outdir / f"port_visits_{self.index:04d}.csv.gz"
        fresh = not path.exists() or path.stat().st_size == 0
        self.fh = gzip.open(path, "at", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=FIELDS)
        if fresh:
            self.writer.writeheader()
        log(f"  -> writing {path.name}")

    def write(self, row: dict):
        if self.fh is None:
            self._open()
        self.writer.writerow(row)
        self.rows += 1
        if self.rows >= SHARD_ROWS:
            self.close()
            self.index += 1
            self.rows = 0

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = self.writer = None


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <work_dir>")
    work = Path(sys.argv[1]).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("GFW_TOKEN", "").strip()
    if not token:
        sys.exit("GFW_TOKEN is not set.  export GFW_TOKEN='<token>'")

    imo_file = work / "imos.txt"
    if not imo_file.exists():
        imo_file.write_text("# one IMO number per line\n9811000\n")
        sys.exit(f"wrote a template to {imo_file} -- fill it in and re-run")
    imos = [l.strip() for l in imo_file.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    log(f"{len(imos):,} IMOs requested")

    ids = resolve_ids(token, imos, work / "vessel_ids.csv")

    ckpt = work / "checkpoint.json"
    done: set[str] = set()
    shard_index = written = 0
    if ckpt.exists():
        c = json.loads(ckpt.read_text())
        done = set(c["done_imos"])
        shard_index, written = c["shard_index"], c["written"]
        log(f"resuming: {len(done):,} vessels done, {written:,} rows written")

    writer = ShardWriter(work, shard_index)
    dropped = collections.Counter()
    requests_made = 0
    t0 = time.time()
    pending = [i for i in imos if i in ids and i not in done]
    log(f"pulling port visits for {len(pending):,} vessels "
        f"({START_DATE} .. {END_DATE})")

    try:
        for n, imo in enumerate(pending, 1):
            offset = 0
            failed = False
            while True:
                d = api_get(token, "events", [
                    ("datasets[0]", PORTVISIT_DS), ("vessels[0]", ids[imo]),
                    ("start-date", START_DATE), ("end-date", END_DATE),
                    ("limit", str(PAGE)), ("offset", str(offset)),
                ])
                requests_made += 1
                if d is None:
                    log(f"  incomplete: IMO {imo} -- will retry on next run")
                    failed = True
                    break
                entries = d.get("entries") or []
                for e in entries:
                    row = flatten(e, imo)
                    ok, why = keep(row)
                    if ok:
                        writer.write(row)
                        written += 1
                    else:
                        dropped[why] += 1
                nxt = d.get("nextOffset")
                if not entries or nxt is None:
                    break
                offset = nxt
                time.sleep(PAUSE)

            if not failed:
                done.add(imo)
            ckpt.write_text(json.dumps({
                "done_imos": sorted(done), "shard_index": writer.index,
                "written": written}))
            if n % 100 == 0:
                el = time.time() - t0
                log(f"{n:,}/{len(pending):,} vessels | {written:,} rows | "
                    f"{requests_made:,} req | {n/max(el,1)*3600:,.0f} vessels/h | "
                    f"dropped {sum(dropped.values()):,}")
            time.sleep(PAUSE)
    finally:
        writer.close()

    log(f"finished: {written:,} rows from {len(done):,} vessels, "
        f"{requests_made:,} requests, {time.time()-t0:,.0f}s")
    if dropped:
        log("dropped: " + ", ".join(f"{k}={v:,}" for k, v in dropped.most_common()))
    log(f"output: {work}")


if __name__ == "__main__":
    main()

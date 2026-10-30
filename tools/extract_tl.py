"""Build the tl extract from the public Ist-Daten archive of opentransportdata.swiss.

    python tools/extract_tl.py 2026-09            # one month, written to data/raw

The monthly archive (about 1.4 GB) holds one CSV per day for all of Switzerland. The script reads the zip's
central directory with HTTP range requests, streams each daily file through zlib, keeps the operator TL
(Transports publics de la région lausannoise) and writes the compact format described in tlpunct/data.py.
Nothing but the tl rows is stored, so a month fits in about 40 MB of gzip.
"""
import csv
import gzip
import io
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

ARCHIVE = "https://archive.opentransportdata.swiss/istdaten/{y}/ist-daten-v2-{y}-{m}.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def entries(url):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        size = int(r.headers["Content-Length"])
    tail = fetch(url, size - 70000, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    cd = fetch(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while cd[p:p + 4] == b"PK\x01\x02":
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        out.append((cd[p + 46:p + 46 + fl].decode(), csize, off))
        p += 46 + fl + el + cl
    return out


def to_sec(s):
    if not s:
        return ""
    t = s.split(" ")[1].split(":")
    return int(t[0]) * 3600 + int(t[1]) * 60 + (int(t[2]) if len(t) > 2 else 0)


def day_rows(url, name, csize, off, trips, stops):
    head = fetch(url, off, off + 29)
    fl, el = struct.unpack("<HH", head[26:30])
    start = off + 30 + fl + el
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + csize - 1}"})
    d = zlib.decompressobj(-15)
    with urllib.request.urlopen(req) as r:
        text = io.TextIOWrapper(_Inflate(r, d), encoding="utf-8", newline="")
        rd = csv.DictReader(text, delimiter=";")
        seq = {}
        for f in rd:
            if f["BETREIBER_ABK"] != "TL":
                continue
            key = (f["BETRIEBSTAG"], f["FAHRT_BEZEICHNER"])
            t = trips.setdefault(key, len(trips))
            q = seq.get(t, 0)
            seq[t] = q + 1
            stops.setdefault(f["BPUIC"], f["HALTESTELLEN_NAME"])
            sa, pa, sd, pd_ = (to_sec(f[k]) for k in ("ANKUNFTSZEIT", "AN_PROGNOSE", "ABFAHRTSZEIT", "AB_PROGNOSE"))
            dd, mm, yy = f["BETRIEBSTAG"].split(".")
            yield [f"{yy}-{mm}-{dd}", t, q, f["LINIEN_TEXT"], f["VERKEHRSMITTEL_TEXT"],
                   int(f["FAELLT_AUS_TF"] == "true"), int(f["ZUSATZFAHRT_TF"] == "true"), f["BPUIC"],
                   sa, (pa - sa) if sa != "" and pa != "" else "", f["AN_PROGNOSE_STATUS"][:4],
                   sd, (pd_ - sd) if sd != "" and pd_ != "" else "", f["AB_PROGNOSE_STATUS"][:4]]


class _Inflate(io.RawIOBase):
    def __init__(self, raw, d):
        self.raw, self.d, self.buf = raw, d, b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            chunk = self.raw.read(1 << 20)
            if not chunk:
                self.buf = self.d.flush()
                if not self.buf:
                    return 0
                break
            self.buf = self.d.decompress(chunk)
        n = min(len(b), len(self.buf))
        b[:n], self.buf = self.buf[:n], self.buf[n:]
        return n


def main(month):
    y, m = month.split("-")
    url = ARCHIVE.format(y=y, m=m)
    OUT.mkdir(parents=True, exist_ok=True)
    trips, stops = {}, {}
    days = entries(url)
    half = (len(days) + 1) // 2
    for part, chunk in enumerate((days[:half], days[half:]), 1):
        with gzip.open(OUT / f"tl_istdaten_{month}_part{part}.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            for name, csize, off in chunk:
                n = 0
                for row in day_rows(url, name, csize, off, trips, stops):
                    w.writerow(row)
                    n += 1
                print(name, n, "tl rows", flush=True)
    with open(OUT / "tl_stops.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stop", "name"])
        w.writerows(sorted(stops.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-09")

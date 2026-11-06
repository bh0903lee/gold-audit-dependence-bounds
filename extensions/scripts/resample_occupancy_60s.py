"""Resample the UCI Room Occupancy Estimation readings to an exact 60-second grid.
Native readings are 30 or 31 s apart, while the windows need an exact lag. Rule (fixed before any interval was
computed): bin = floor(local wall-clock seconds-of-day / 60), i.e. the local clock minute; within a bin
  count = max(Room_Occupancy_Count), S6_PIR = max(S6_PIR), S7_PIR = max(S7_PIR)  (an OR over the readings in the bin).
Minutes with no reading are omitted (no interpolation), so a gap breaks window continuity.
Usage: python -I resample_occupancy_60s.py Occupancy_Estimation.csv occupancy_60s.csv"""
import csv, hashlib, sys
from collections import OrderedDict
src, out = sys.argv[1], sys.argv[2]
bins = OrderedDict(); n = 0
with open(src, newline='') as f:
    for row in csv.DictReader(f):
        n += 1; d = row['Date'].replace('/', '-'); h, m, s = row['Time'].split(':')
        key = (d, int(h), int(m))
        cur = bins.get(key, [0, 0, 0, 0])
        cur = [max(cur[0], int(row['Room_Occupancy_Count'])), max(cur[1], int(row['S6_PIR'])), max(cur[2], int(row['S7_PIR'])), cur[3] + 1]
        bins[key] = cur
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['timestamp', 'unit', 'count', 'S6_PIR', 'S7_PIR', 'readings_in_bin'])
    for (d, h, m), (c, p6, p7, k) in sorted(bins.items()): w.writerow([f'{d}T{h:02d}:{m:02d}:00', 'room_6x4.6m', c, p6, p7, k])
b = open(out, 'rb').read()
print(dict(input_rows=n, bins=len(bins), readings_per_bin={k: v for k, v in sorted(__import__('collections').Counter(v[3] for v in bins.values()).items())}, out_sha256=hashlib.sha256(b).hexdigest(), out_bytes=len(b)))

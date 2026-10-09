import argparse
import bz2
import glob
import io
import json
import os
import shutil
import zipfile
import numpy as np
from PIL import Image

# CED (Color Event Camera Dataset, CVPRW'19): Color DAVIS346, events and frames from the same 346x260 pixels behind an RGBG
# Bayer filter. Input: the per-category zips (or folders) of ROS1 bags: dvs_msgs/EventArray events and sensor_msgs/Image frames,
# image_raw (mono8 Bayer mosaic, linear) and image_color (demosaiced + sRGB gamma, for viewing). Bags are read straight from
# the zip record by record (no ROS, no index): bags stored one topic after another, with a broken index or cut off convert.
# Frames: image_raw demosaiced with the rggb pattern (image_raw vs image_color picked rggb in all 68 bags; re-checked per bag).
# No official split: every 5th sequence of each category (driving, indoors, people, simple) is test.
URL = "https://rpg.ifi.uzh.ch/CED.html"
EV = np.dtype([("x", "<u2"), ("y", "<u2"), ("s", "<u4"), ("ns", "<u4"), ("p", "u1")])  # dvs_msgs/Event, ROS1 wire format

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="CED zips of ROS bags (or folders) -> <out>/{train,test}/<sequence>/")
ap.add_argument("--src", nargs="+", default=glob.glob(os.path.join(HERE, "*.zip")) or [HERE], help="zip file(s) or folder(s)")
ap.add_argument("--out", default="data/ced")
ap.add_argument("--bayer", default="rggb", help="color of the top-left 2x2 pixels of image_raw, in reading order")
ap.add_argument("--offset-ms", type=float, default=0.0, help="added to frame times")
ap.add_argument("--dry", action="store_true", help="only list the bags and their split")
a = ap.parse_args()
u32 = lambda b, o=0: int.from_bytes(b[o:o + 4], "little")


def fields(b):
    h, p = {}, 0
    while p < len(b):
        k, _, v = b[p + 4:p + 4 + u32(b, p)].partition(b"=")
        h[k.decode()], p = v, p + 4 + u32(b, p)
    return h


def records(f):
    # ROS1 v2.0 record: header length, header fields (op=..., conn=...), data length, data; yields (header, data, whole)
    while len(n := f.read(4)) == 4 and u32(n) < 1 << 20:
        h, n = fields(f.read(u32(n))), f.read(4)
        d = f.read(u32(n)) if len(n) == 4 else b""
        yield h, d, len(n) == 4 and len(d) == u32(n)


def unpack(comp, d, info):
    try:
        if comp == "none":
            return d
        if comp == "bz2":
            return bz2.BZ2Decompressor().decompress(d)  # a cut-off chunk gives the part that is there
        import lz4.frame
        return lz4.frame.LZ4FrameDecompressor().decompress(d)
    except Exception:
        info["bad"] += 1
        return b""


def messages(f, info):
    # (topic, type, raw message) of every complete message, in file order
    if f.read(13) != b"#ROSBAG V2.0\n":
        raise ValueError("not a ROS1 v2.0 bag")
    conns = {}
    for h, d, whole in records(f):
        op, info["whole"] = (h.get("op") or b"\0")[0], whole
        info["index"] = info["index"] or op == 6  # chunk info records sit in the index at the end
        for h2, d2, whole2 in records(io.BytesIO(unpack(h.get("compression", b"none").decode(), d, info))) if op == 5 else [(h, d, whole)]:
            if not whole2:
                break
            if h2.get("op") == b"\x07":
                conns[u32(h2["conn"])] = (h2["topic"].decode(), fields(d2)["type"].decode())
            elif h2.get("op") == b"\x02" and u32(h2["conn"]) in conns:
                yield *conns[u32(h2["conn"])], d2


def events(raw):
    # dvs_msgs/EventArray: header (seq, stamp, frame_id), height, width, events[]
    o = 16 + u32(raw, 12) + 8
    return np.frombuffer(raw, EV, u32(raw, o), o + 4)


def image(raw):
    # sensor_msgs/Image: header (seq, stamp, frame_id), height, width, encoding, is_bigendian, step, data[]
    sec, nsec, o = u32(raw, 4), u32(raw, 8), 16 + u32(raw, 12)
    h, w, n = u32(raw, o), u32(raw, o + 4), u32(raw, o + 8)
    enc, o = raw[o + 12:o + 12 + n].decode(), o + 12 + n
    big, step, size = raw[o], u32(raw, o + 1), 2 if "16" in enc else 1
    ch = step // (w * size)
    img = np.frombuffer(raw, np.uint8, h * step, o + 9).reshape(h, step)[:, :w * ch * size].copy()
    if size == 2:  # 16-bit -> 8-bit
        img = (img.view(">u2" if big else "<u2") >> 8).astype(np.uint8)
    return sec, nsec, enc, img.reshape(h, w, ch)


def demosaic(m, pattern):
    # bilinear demosaic; measured sites keep their value. pattern = colors of the top-left 2x2 block in reading order
    H, W = m.shape
    rgb, m = np.zeros((H, W, 3), np.float32), m.astype(np.float32)
    for c, name in enumerate("rgb"):
        k = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]] if name == "g" else [[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
        conv = lambda v: sum(k[i, j] * np.pad(v, 1)[i:i + H, j:j + W] for i in range(3) for j in range(3))
        mask = np.zeros((H, W), np.float32)
        for q, ch in enumerate(pattern):
            if ch == name:
                mask[q // 2::2, q % 2::2] = 1
        rgb[..., c] = conv(m * mask) / np.maximum(conv(mask), 1e-6)
    return rgb.clip(0, 255).astype(np.uint8)


def bayer_scores(raw, color):
    # rank correlation of image_raw sites with the image_color channel each pattern assigns them: the right one is ~1
    rt = np.array([s + n * 1e-9 for s, n, _, _ in raw])
    pairs = [(raw[j][3][..., 0], img[..., 2::-1] if enc.lower().startswith("bgr") else img[..., :3]) for s, n, enc, img in color
             for j in [np.abs(rt - s - n * 1e-9).argmin()] if abs(rt[j] - s - n * 1e-9) < 2e-3][:10]
    rank = lambda v: np.argsort(np.argsort(v.ravel())).astype(np.float64)
    return {p: np.mean([np.corrcoef(rank(r[q // 2::2, q % 2::2]), rank(c[q // 2::2, q % 2::2, "rgb".index(ch)]))[0, 1]
                        for r, c in pairs for q, ch in enumerate(p)]) for p in ("rggb", "grbg", "gbrg", "bggr")} if pairs else {}


def convert(S, member, dst):
    info, ev, raw, color, n_color = {"bad": 0, "index": False, "whole": True}, [], [], [], 0
    with S.open(member) as f:
        for topic, typ, msg in messages(f, info):
            name = topic.rsplit("/", 1)[-1]
            if typ.endswith("EventArray"):
                ev.append(events(msg))
            elif typ == "sensor_msgs/Image" and name == "image_raw":
                raw.append(image(msg))
            elif typ == "sensor_msgs/Image" and name == "image_color":
                n_color += 1
                if n_color % 20 == 1:  # every 20th color frame is enough to check the Bayer pattern
                    color.append(image(msg))
    print(f"{os.path.basename(member)}: " + ", ".join(
        ["read whole" if info["whole"] and info["index"] else "! end of file missing or damaged (no index found), read what is there"]
        + [f"! {info['bad']} unreadable chunks skipped"] * bool(info["bad"])), flush=True)
    if not ev or len(raw) < 2:
        raise ValueError(f"{sum(map(len, ev))} events, {len(raw)} image_raw frames")
    enc = raw[0][2].lower()
    pat = enc[6:10] if enc.startswith("bayer_") else a.bayer  # the encoding names the pattern when it can
    sc = bayer_scores(raw, color)
    best = sorted(sc, key=sc.get, reverse=True)
    if sc:
        print(f"  Bayer check: {pat} {sc[pat]:.3f}, best other {next(p for p in best if p != pat)} "
              f"{max(v for p, v in sc.items() if p != pat):.3f}" + (f"  ! {best[0]} fits better" if sc[best[0]] > sc[pat] + 0.01 else ""))
    ev = np.concatenate(ev)
    base = int(ev["s"].min())  # whole seconds removed before going to float: keeps sub-microsecond precision
    t = (ev["s"] - base).astype(np.float64) + ev["ns"] * 1e-9
    ts = np.array([s - base + n * 1e-9 for s, n, _, _ in raw]) + a.offset_ms * 1e-3
    keep = (ts >= t.min()) & (ts <= t.max())  # frames inside the event stream
    if keep.sum() < 2:
        raise ValueError("fewer than 2 frames inside the event stream")
    frames, ts, o = [f for f, k in zip(raw, keep) if k], ts[keep], np.argsort(t, kind="stable")
    os.makedirs(os.path.join(dst, "frames"))
    for k, (_, _, _, img) in enumerate(frames):
        Image.fromarray(demosaic(img[..., 0], pat)).save(os.path.join(dst, "frames", f"{k:06d}.png"))
    np.save(os.path.join(dst, "frame_ts.npy"), ts - ts[0])
    np.save(os.path.join(dst, "ev_t.npy"), t[o] - ts[0])
    np.save(os.path.join(dst, "ev_x.npy"), ev["x"][o].astype(np.int16))
    np.save(os.path.join(dst, "ev_y.npy"), ev["y"][o].astype(np.int16))
    np.save(os.path.join(dst, "ev_p.npy"), np.where(ev["p"][o] > 0, 1, -1).astype(np.int8))
    open(os.path.join(dst, "bayer.txt"), "w").write(pat + "\n")
    open(os.path.join(dst, "done"), "w").close()
    H, W = frames[0][3].shape[:2]
    print(f"  -> {dst}: {len(frames)}/{len(raw)} frames {W}x{H} @ {1 / np.median(np.diff(ts)):.1f} fps, {len(t)} events", flush=True)


class Src:
    # a zip or a folder, read the same way
    def __init__(self, path):
        self.path, self.z = path, zipfile.ZipFile(path) if os.path.isfile(path) else None
        self.names = self.z.namelist() if self.z else [os.path.relpath(os.path.join(d, f), path)
                                                       for d, _, fs in os.walk(path, followlinks=True) for f in fs]

    def open(self, name):
        return self.z.open(name) if self.z else open(os.path.join(self.path, name), "rb", buffering=1 << 24)


bags = {}  # bag name -> (source, member); the first one found wins
for S in map(Src, a.src):
    for n in S.names:
        if n.endswith(".bag") and "__MACOSX" not in n:
            bags.setdefault(os.path.basename(n)[:-4], (S, n))
cats = {}
for name in sorted(bags):
    cats.setdefault(name.split("_")[0], []).append(name)
test = {name for group in cats.values() for name in group[2::5]}
os.makedirs(a.out, exist_ok=True)
json.dump(dict(name="CED", paper="CED: Color Event Camera Dataset (CVPRW 2019)", url=URL,
               events="Color DAVIS346, same pixels as the frames", frames=f"image_raw demosaiced ({a.bayer}), linear, 346x260",
               cfa=a.bayer, splits="every 5th sequence of each category is test", converter="CED/ced.py"),
          open(os.path.join(a.out, "info.json"), "w"), indent=1)
print(f"{len(bags)} bags, {len(test)} test")
for name, (S, member) in sorted(bags.items()):
    dst = os.path.join(a.out, "test" if name in test else "train", name)
    if os.path.exists(os.path.join(dst, "done")):
        continue
    if a.dry:
        print(f"  {os.path.basename(os.path.dirname(dst))}: {name}")
        continue
    shutil.rmtree(dst, ignore_errors=True)
    try:
        convert(S, member, dst)
    except Exception as e:
        print(f"  ! {name}: skipped ({type(e).__name__}: {e})", flush=True)

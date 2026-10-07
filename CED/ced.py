import argparse
import bz2
import glob
import mmap
import os
import numpy as np
from PIL import Image

# CED (Color Event Camera Dataset, CVPRW'19): Color DAVIS346, events and frames from the same 346x260 pixels; ROS1 bags with
# dvs_msgs/EventArray events, sensor_msgs/Image frames (image_raw: sensor output, image_color: demosaiced + sRGB gamma).
# Frames come from image_raw when it is a Bayer mosaic (linear light, pattern kept in bayer.txt for the event physics),
# else from image_color. No official split: every 5th sequence of each category (driving, indoors, people, simple) is test.
# Bags are read record by record (no ROS, no rosbags, no index): bags with a broken index or a cut-off end still convert.
EV = np.dtype([("x", "<u2"), ("y", "<u2"), ("s", "<u4"), ("ns", "<u4"), ("p", "u1")])  # dvs_msgs/Event, ROS1 wire format

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="CED bags (in this directory) -> evtween layout: <out>/{train,test}/<sequence>/")
ap.add_argument("--out", default="data/ced")
ap.add_argument("--raw", default=HERE, help="folder holding the .bag files")
ap.add_argument("--bayer", default=None, help="pattern of a mono8 image_raw, e.g. rggb (read from the encoding if given there)")
ap.add_argument("--offset-ms", type=float, default=0.0, help="added to frame times (e.g. half the exposure, after check.py)")
ap.add_argument("--dry", action="store_true", help="only print topics, message counts, time spans, image encodings (from the bag index)")
a = ap.parse_args()
u32 = lambda b, o=0: int.from_bytes(b[o:o + 4], "little")


def fields(buf, p, end):
    h = {}
    while p < end:
        n = u32(buf, p)
        k, _, v = bytes(buf[p + 4:p + 4 + n]).partition(b"=")
        h[k.decode()], p = v, p + 4 + n
    return h


def records(buf, o=0):
    # ROS1 v2.0 record: header length, header fields (op=..., conn=..., ...), data length, data; yields (header, data, end)
    while o + 8 <= len(buf):
        hl = u32(buf, o)
        if hl > 1 << 20 or o + 8 + hl > len(buf):  # header cut off (or garbage)
            return
        h, start = fields(buf, o + 4, o + 4 + hl), o + 8 + hl
        o = start + u32(buf, start - 4)
        yield h, buf[start:o], o


def unpack(comp, data, info):
    try:
        if comp == "none":
            return data
        if comp == "bz2":
            return bz2.BZ2Decompressor().decompress(data)  # a cut-off chunk gives the part that is there
        import lz4.frame
        return lz4.frame.LZ4FrameDecompressor().decompress(data)
    except Exception:
        info["bad"] += 1
        return b""


def messages(bag, info):
    # (topic, type, bag time, raw message) of every complete message, in file order
    conns = {}
    with open(bag, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        if m[:13] != b"#ROSBAG V2.0\n":
            raise ValueError("not a ROS1 v2.0 bag")
        info.update(size=len(m), end=13)
        for h, data, end in records(m, 13):
            op = h.get("op", b"\0")[0]
            if op == 3:
                info["index"] = int.from_bytes(h["index_pos"], "little")
            cd = unpack(h["compression"].decode(), data, info) if op == 5 else data
            for h2, d2, e2 in records(cd) if op == 5 else [(h, data, end)]:
                if e2 > len(cd):
                    break
                if h2.get("op") == b"\x07":
                    conns[u32(h2["conn"])] = (h2["topic"].decode(), fields(d2, 0, len(d2))["type"].decode())
                elif h2.get("op") == b"\x02" and u32(h2["conn"]) in conns:
                    yield *conns[u32(h2["conn"])], u32(h2["time"]) + u32(h2["time"], 4) * 1e-9, d2
            if end <= len(m):
                info["end"] = end


def events(raw):
    # dvs_msgs/EventArray: header (seq, stamp, frame_id), height, width, events[]
    o = 16 + u32(raw, 12) + 8
    return np.frombuffer(raw, EV, u32(raw, o), o + 4)


def image(raw):
    # sensor_msgs/Image: header (seq, stamp, frame_id), height, width, encoding, is_bigendian, step, data[]
    sec, nsec, o = u32(raw, 4), u32(raw, 8), 16 + u32(raw, 12)
    h, w, n = u32(raw, o), u32(raw, o + 4), u32(raw, o + 8)
    enc, o = bytes(raw[o + 12:o + 12 + n]).decode(), o + 12 + n
    big, step, size = raw[o], u32(raw, o + 1), 2 if "16" in enc else 1
    ch = step // (w * size)
    img = np.frombuffer(raw, np.uint8, h * step, o + 9).reshape(h, step)[:, :w * ch * size].copy()
    if size == 2:  # 16-bit -> 8-bit
        img = (img.view(">u2" if big else "<u2") >> 8).astype(np.uint8)
    return sec, nsec, enc, img.reshape(h, w, ch)


def demosaic(m, pattern):
    # bilinear demosaic of a Bayer mosaic; pattern = colors of the top-left 2x2 block in reading order, e.g. "rggb"
    H, W = m.shape
    rgb, m = np.zeros((H, W, 3), np.float32), m.astype(np.float32)
    for c, name in enumerate("rgb"):
        k = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]] if name == "g" else [[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
        conv = lambda v: sum(k[i, j] * np.pad(v, 1)[i:i + H, j:j + W] for i in range(3) for j in range(3))
        mask = np.zeros((H, W), np.float32)
        for q, ch in enumerate(pattern):
            if ch == name:
                mask[q // 2::2, q % 2::2] = 1
        rgb[..., c] = conv(m * mask) / np.maximum(conv(mask), 1e-6)  # measured sites keep their value
    return rgb.clip(0, 255).astype(np.uint8)


def summary(bag):
    # whole bag from the index at the end of the file (no data chunk is read): every topic, its message count and time span,
    # whether the file is stored in time order, and the encoding of the first image of each image topic
    stamp = lambda b: u32(b) + u32(b, 4) * 1e-9
    with open(bag, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        ip, conns, chunks = int.from_bytes(next(records(m, 13))[0].get("index_pos", b""), "little"), {}, []
        try:
            for h, d, _ in records(m, ip) if 0 < ip < len(m) else []:
                if h["op"] == b"\x07":
                    conns[u32(h["conn"])] = (h["topic"].decode(), fields(d, 0, len(d))["type"].decode())
                elif h["op"] == b"\x06":
                    chunks.append((int.from_bytes(h["chunk_pos"], "little"), stamp(h["start_time"]), stamp(h["end_time"]),
                                   {u32(d, 8 * i): u32(d, 8 * i + 4) for i in range(u32(h["count"]))}))
        except Exception:
            chunks = []
        if not chunks:
            print(f"{os.path.basename(bag)}: {len(m) / 2 ** 30:.2f} GB, index missing or damaged (run without --dry to check it)")
            return
        chunks.sort()
        starts = np.array([c[1] for c in chunks])
        back, T0 = (np.maximum.accumulate(starts) - starts).max(), starts.min()
        print(f"{os.path.basename(bag)}: {len(m) / 2 ** 30:.2f} GB, {len(chunks)} chunks, "
              + ("stored in time order" if back < 1 else f"NOT stored in time order (jumps back {back:.0f} s)"))
        for k, (topic, typ) in sorted(conns.items(), key=lambda kv: kv[1]):
            ch = [c for c in chunks if k in c[3]]
            if not ch:
                continue
            line = (f"  {topic:20s} {typ:20s} {sum(c[3][k] for c in ch):6d} msgs, "
                    f"{min(c[1] for c in ch) - T0:6.1f} to {max(c[2] for c in ch) - T0:6.1f} s")
            if typ == "sensor_msgs/Image":
                h, d, _ = next(records(m, ch[0][0]))
                d2 = next(d2 for h2, d2, _ in records(unpack(h["compression"].decode(), d, {"bad": 0}))
                          if h2.get("op") == b"\x02" and u32(h2["conn"]) == k)
                _, _, enc, img = image(d2)
                line += f", {enc} {img.shape[1]}x{img.shape[0]}x{img.shape[2]}"
            print(line)


def detect(raw, color):
    # Bayer pattern of a mono image_raw: the one whose measured sites match image_color best (rank correlation: any gamma)
    rt = np.array([s + n * 1e-9 for s, n, _, _ in raw])
    pairs = [(raw[j][3][..., 0], rgb(enc, img, None)) for s, n, enc, img in color
             for j in [np.abs(rt - s - n * 1e-9).argmin()] if abs(rt[j] - s - n * 1e-9) < 2e-3]  # same frame: times within 2 ms
    pairs = pairs[::max(1, len(pairs) // 10)][:10]
    rank = lambda v: np.argsort(np.argsort(v.ravel())).astype(np.float64)
    score = lambda p: np.mean([np.corrcoef(rank(r[q // 2::2, q % 2::2]), rank(c[q // 2::2, q % 2::2, "rgb".index(ch)]))[0, 1]
                               for r, c in pairs for q, ch in enumerate(p)])
    scores = {p: score(p) for p in ("rggb", "grbg", "gbrg", "bggr")} if pairs else {}
    top = sorted(scores.values())[-2:] if scores else [0, 0]
    return max(scores, key=scores.get) if top[1] > 0.8 and top[1] - top[0] > 0.1 else None, scores  # unclear: image_color


def read(bag, every=20):
    # keeps every `every`-th image_color frame (enough to find the Bayer pattern); reads again keeping all if frames come from it
    info, ev, ims, n = {"bad": 0}, [], {}, 0
    for topic, typ, t, raw in messages(bag, info):
        name = topic.rsplit("/", 1)[-1]
        if typ.endswith("EventArray"):
            ev.append(events(raw))
        elif typ == "sensor_msgs/Image" and name in ("image_raw", "image_color"):
            n += name == "image_color"
            if name == "image_raw" or (n - 1) % every == 0:
                ims.setdefault(name, []).append(image(raw))
    size, end, index = info.get("size", 1), info.get("end", 0), info.get("index", 0)
    every > 1 and print(f"{os.path.basename(bag)}: {size / 2 ** 30:.2f} GB, "
          f"index {'missing' if index == 0 else 'beyond the file end' if index >= size else 'present'}"
          + ("" if end >= size else f", index damaged (all data read)" if 0 < index <= end
             else f", ! file cut off: read up to {100 * end / size:.1f}%")
          + (f", ! {info['bad']} unreadable chunks skipped" if info["bad"] else ""))
    if not ev:
        raise ValueError("no events")
    raw_enc = ims["image_raw"][0][2].lower() if "image_raw" in ims else ""
    pattern = raw_enc.split("_")[1][:4] if raw_enc.startswith("bayer_") else a.bayer if raw_enc else None
    if every > 1 and raw_enc and not pattern and "image_color" in ims and ims["image_raw"][0][3].shape[2] == 1:
        pattern, scores = detect(ims["image_raw"], ims["image_color"])
        print("  Bayer pattern from image_raw vs image_color: "
              + (", ".join(f"{p} {v:.3f}" for p, v in scores.items()) if scores else "no frames with matching times")
              + ("" if pattern else " -> unclear, frames from image_color"))
    name = "image_raw" if pattern else "image_color"
    if name == "image_color" and every > 1:
        return read(bag, 1)
    if name not in ims:
        raise ValueError(f"no usable frames (image_raw encoding {raw_enc or 'none'}, no image_color); try --bayer")
    return np.concatenate(ev), ims[name], pattern


def rgb(enc, img, pattern):
    if pattern:
        return demosaic(img[..., 0], pattern)
    return img[..., [0, 0, 0]] if img.shape[2] == 1 else img[..., 2::-1] if enc.lower().startswith("bgr") else img[..., :3]


bags = sorted(glob.glob(os.path.join(a.raw, "**", "*.bag"), recursive=True))
cats = {}
for b in bags:
    cats.setdefault(os.path.basename(b).split("_")[0], []).append(b)
test = {b for group in cats.values() for b in group[2::5]}
print(f"{len(bags)} bags, {len(test)} test")
for bag in bags:
    try:
        out = summary(bag) if a.dry else read(bag)
    except Exception as e:
        print(f"{os.path.basename(bag)}: skipped ({type(e).__name__}: {e})")
        continue
    if out is None:
        continue
    ev, frames, pattern = out
    base = int(ev["s"].min())  # whole seconds removed before going to float: keeps sub-microsecond precision
    t = (ev["s"] - base).astype(np.float64) + ev["ns"] * 1e-9
    ts = np.array([sec - base + nsec * 1e-9 for sec, nsec, _, _ in frames]) + a.offset_ms * 1e-3
    keep = (ts >= t.min()) & (ts <= t.max())  # frames inside the event stream
    if keep.sum() < 2:
        print(f"{os.path.basename(bag)}: skipped (fewer than 2 frames inside the event stream)")
        continue
    n_all, frames, ts = len(frames), [f for f, k in zip(frames, keep) if k], ts[keep]
    t0, o = ts[0], np.argsort(t, kind="stable")
    dst = os.path.join(a.out, "test" if bag in test else "train", os.path.splitext(os.path.basename(bag))[0])
    os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
    for k, (_, _, enc, img) in enumerate(frames):
        Image.fromarray(rgb(enc, img, pattern)).save(os.path.join(dst, "frames", f"{k:06d}.png"))
    np.save(os.path.join(dst, "frame_ts.npy"), ts - t0)
    np.save(os.path.join(dst, "ev_t.npy"), t[o] - t0)
    np.save(os.path.join(dst, "ev_x.npy"), ev["x"][o].astype(np.int16))
    np.save(os.path.join(dst, "ev_y.npy"), ev["y"][o].astype(np.int16))
    np.save(os.path.join(dst, "ev_p.npy"), np.where(ev["p"][o] > 0, 1, -1).astype(np.int8))
    if pattern:
        open(os.path.join(dst, "bayer.txt"), "w").write(pattern + "\n")
    H, W = frames[0][3].shape[:2]
    print(f"  -> {dst}: {len(frames)}/{n_all} frames {W}x{H} @ {1 / np.median(np.diff(ts)):.1f} fps, {len(t)} events, "
          f"frames from {'image_raw (' + pattern + ')' if pattern else 'image_color'}")

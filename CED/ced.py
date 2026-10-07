import argparse
import glob
import os
from pathlib import Path
import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader

# CED (Color Event Camera Dataset, CVPRW'19): Color DAVIS346, events and frames from the same 346x260 pixels; ROS1 bags with
# dvs_msgs/EventArray events, sensor_msgs/Image frames (image_raw: sensor output, image_color: demosaiced + sRGB gamma).
# Frames come from image_raw when it is a Bayer mosaic (linear light, pattern kept in bayer.txt for the event physics),
# else from image_color. No official split: every 5th sequence of each category (driving, indoors, people, simple) is test.
EV = np.dtype([("x", "<u2"), ("y", "<u2"), ("s", "<u4"), ("ns", "<u4"), ("p", "u1")])  # dvs_msgs/Event, ROS1 wire format

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="CED bags (in this directory) -> evtween layout: <out>/{train,test}/<sequence>/")
ap.add_argument("--out", default="data/ced")
ap.add_argument("--raw", default=HERE, help="folder holding the .bag files")
ap.add_argument("--bayer", default=None, help="pattern of a mono8 image_raw, e.g. rggb (read from the encoding if given there)")
ap.add_argument("--offset-ms", type=float, default=0.0, help="added to frame times (e.g. half the exposure, after check.py)")
ap.add_argument("--dry", action="store_true", help="only print topics, image encoding and sizes")
a = ap.parse_args()


def events(raw):
    # dvs_msgs/EventArray: header (seq, stamp, frame_id), height, width, events[]
    raw = bytes(raw)
    o = 16 + int.from_bytes(raw[12:16], "little") + 8
    return np.frombuffer(raw, EV, int.from_bytes(raw[o:o + 4], "little"), o + 4)


def demosaic(m, pattern):
    # bilinear demosaic of a Bayer mosaic; pattern = colors of the top-left 2x2 block in reading order, e.g. "rggb"
    H, W = m.shape
    rgb, m = np.zeros((H, W, 3), np.float32), m.astype(np.float32)
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
    conv = lambda v: sum(k[i, j] * np.pad(v, 1)[i:i + H, j:j + W] for i in range(3) for j in range(3))
    for c, name in enumerate("rgb"):
        mask = np.zeros((H, W), np.float32)
        for q, ch in enumerate(pattern):
            if ch == name:
                mask[q // 2::2, q % 2::2] = 1
        rgb[..., c] = conv(m * mask) / np.maximum(conv(mask), 1e-6)
    return rgb.clip(0, 255).astype(np.uint8)


def read(bag):
    with AnyReader([Path(bag)]) as r:
        ev_c = [c for c in r.connections if c.msgtype.endswith("EventArray")]
        im_c = {c.topic.rsplit("/", 1)[-1]: c for c in r.connections if c.msgtype.endswith("msg/Image")}
        if a.dry:
            print(bag, [(c.topic, c.msgtype, c.msgcount) for c in r.connections])
        ev = None if a.dry else np.concatenate([events(raw) for _, _, raw in r.messages(connections=ev_c)])
        frames, ts, pattern = [], [], None
        for name in ("image_raw", "image_color"):
            if name not in im_c:
                continue
            for c, _, raw in r.messages(connections=[im_c[name]]):
                msg = r.deserialize(raw, c.msgtype)
                img = np.asarray(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
                enc = msg.encoding.lower()
                pat = enc.split("_")[1][:4] if enc.startswith("bayer_") else (a.bayer if name == "image_raw" else None)
                if name == "image_raw" and pat is None:
                    break  # mono raw without a known pattern: use image_color
                if a.dry:
                    print(f"  {name}: encoding {msg.encoding}, {msg.width}x{msg.height}, {im_c[name].msgcount} frames")
                    return None
                frames.append(demosaic(img[..., 0], pat) if pat else (img[..., ::-1] if enc.startswith("bgr") else img[..., :3]))
                ts.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 + a.offset_ms * 1e-3)
                pattern = pat
            if frames:
                break
    return ev, frames, np.array(ts), pattern


bags = sorted(glob.glob(os.path.join(a.raw, "**", "*.bag"), recursive=True))
cats = {}
for b in bags:
    cats.setdefault(os.path.basename(b).split("_")[0], []).append(b)
test = {b for group in cats.values() for b in group[2::5]}
print(f"{len(bags)} bags, {len(test)} test")
for bag in bags:
    out = read(bag)
    if out is None:
        continue
    ev, frames, ts, pattern = out
    t = ev["s"].astype(np.float64) + ev["ns"] * 1e-9
    keep = (ts >= t.min()) & (ts <= t.max())  # frames inside the event stream
    frames, ts = [f for f, k in zip(frames, keep) if k], ts[keep]
    t0, o = ts[0], np.argsort(t, kind="stable")
    name = os.path.splitext(os.path.basename(bag))[0]
    dst = os.path.join(a.out, "test" if bag in test else "train", name)
    os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
    for k, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(dst, "frames", f"{k:06d}.png"))
    np.save(os.path.join(dst, "frame_ts.npy"), ts - t0)
    np.save(os.path.join(dst, "ev_t.npy"), t[o] - t0)
    np.save(os.path.join(dst, "ev_x.npy"), ev["x"][o].astype(np.int16))
    np.save(os.path.join(dst, "ev_y.npy"), ev["y"][o].astype(np.int16))
    np.save(os.path.join(dst, "ev_p.npy"), np.where(ev["p"][o] > 0, 1, -1).astype(np.int8))
    if pattern:
        open(os.path.join(dst, "bayer.txt"), "w").write(pattern + "\n")
    H, W = frames[0].shape[:2]
    print(f"  -> {dst}: {len(frames)} frames {W}x{H} @ {1 / np.median(np.diff(ts)):.1f} fps, {len(t)} events, "
          f"frames from {'image_raw (' + pattern + ')' if pattern else 'image_color'}")

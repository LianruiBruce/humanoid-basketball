"""Cut the ~30 s demo video and the README GIF from recorded clips (no Isaac Sim needed).

    python scripts/make_demo.py
"""

import os

import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "media")
W, H, FPS = 1280, 720, 50
BG = (18, 20, 24)


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def card(lines, seconds):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = H // 2 - sum(s for _, s, _ in lines) // 2 - 20
    for text, size, color in lines:
        f = font(size, bold=size >= 40)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=color)
        y += int(size * 1.5)
    return [np.array(im)] * int(seconds * FPS)


def clip(path, start, seconds, caption, sub=None):
    frames = iio.imread(path)
    n = int(seconds * FPS)
    out = []
    for f in frames[start:start + n]:
        im = Image.fromarray(f)
        im.thumbnail((W, H))
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        d = ImageDraw.Draw(canvas)
        d.rectangle((0, H - 70, W, H), fill=(0, 0, 0))
        d.text((24, H - 62), caption, font=font(30, bold=True), fill=(255, 255, 255))
        if sub:
            d.text((24, H - 26), sub, font=font(18), fill=(190, 190, 190))
        out.append(np.array(canvas))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    o = lambda *p: os.path.join(ROOT, "outputs", *p)  # noqa: E731
    frames = []
    frames += card([("Humanoid Basketball Shooting", 56, (255, 255, 255)),
                    ("RL + human motion prior on a Unitree G1", 30, (210, 210, 210)),
                    ("regulation ball & hoop  |  real motor limits  |  NVIDIA Isaac Lab", 22, (150, 150, 150))], 2.5)
    frames += clip(o("g1_amp_free_throw.mp4"), 0, 6.0, "Free throw, 4.19 m",
                   "87% simulated make rate (2,000 shots)")
    frames += clip(o("g1_amp_three_pointer.mp4"), 0, 6.0, "Three-pointer, 6.75 m",
                   "87% simulated make rate (2,000 shots)")
    frames += clip(o("mocap", "g1_124_04_replay.mp4"), 0, 3.0, "Human reference: CMU mocap retargeted to the G1",
                   "segment-direction matching + batched DLS IK, 0.6 cm error")
    frames += clip(o("g1_before_after_amp.mp4"), 0, 6.0, "Adversarial Motion Prior: before vs. after",
                   "sideways fling  ->  human-like shot with follow-through, no accuracy loss")
    frames += card([("Results (simulation)", 44, (255, 255, 255)),
                    ("Clean sim: 84-90% make rate from 2 m to the 3-point line", 26, (220, 220, 220)),
                    ("With latency + noise + dynamics randomization:", 26, (220, 220, 220)),
                    ("no-DR policy 0%   ->   history + asymmetric critic 49% free throws", 26, (255, 200, 120))], 3.0)
    path = os.path.join(OUT, "demo.mp4")
    imageio.mimsave(path, frames, fps=FPS, quality=7, macro_block_size=8)
    print(f"wrote {path}: {len(frames) / FPS:.1f} s")

    # README GIF: a free throw, every 2nd frame, 25 fps
    src = iio.imread(o("g1_amp_free_throw.mp4"))[0:180:2]
    gif = []
    for f in src:
        im = Image.fromarray(f).resize((480, 270), Image.LANCZOS)
        gif.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    gpath = os.path.join(OUT, "free_throw.gif")
    gif[0].save(gpath, save_all=True, append_images=gif[1:], duration=40, loop=0, optimize=True)
    print(f"wrote {gpath}: {os.path.getsize(gpath) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

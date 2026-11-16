import sys, os, math, numpy as np, torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
torch.set_num_threads(4)
MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
proc = SegformerImageProcessor.from_pretrained(MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
# ADE20K: 0 wall, 3 floor, 5 ceiling, 8 windowpane, 14 door, 22 painting, 27 mirror
GROUPS = {"wall": [0], "ceiling": [5], "floor": [3, 28], "opening": [8, 14, 58]}  # 28 rug, 58 screen door
OW, OH = 2048, 1024
V = 768  # view size

def dirs_equirect(W, H):
    u = (np.arange(W) + .5) / W; v = (np.arange(H) + .5) / H
    phi = u * 2 * np.pi; th = v * np.pi
    P, T = np.meshgrid(phi, th)
    return np.stack([np.cos(P) * np.sin(T), np.cos(T), np.sin(P) * np.sin(T)], -1)  # x,y,z (y up)

def view_basis(yaw, pitch):
    # forward direction using same convention as app: lon=yaw (deg), lat=pitch
    p = math.radians(yaw); t = math.radians(90 - pitch)
    f = np.array([math.cos(p) * math.sin(t), math.cos(t), math.sin(p) * math.sin(t)])
    up0 = np.array([0, 1, 0.])
    r = np.cross(f, up0); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u

def render_view(eq, yaw, pitch, fov):
    H, W = eq.shape[:2]
    f, r, u = view_basis(yaw, pitch)
    k = math.tan(math.radians(fov) / 2)
    xs = (np.arange(V) + .5) / V * 2 - 1
    X, Y = np.meshgrid(xs, -xs)
    d = f[None, None] + X[..., None] * k * r[None, None] + Y[..., None] * k * u[None, None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    phi = np.arctan2(d[..., 2], d[..., 0]) % (2 * np.pi); th = np.arccos(np.clip(d[..., 1], -1, 1))
    px = (phi / (2 * np.pi) * W).astype(int) % W; py = np.clip((th / np.pi * H).astype(int), 0, H - 1)
    return eq[py, px]

def seg(img):
    inp = proc(images=Image.fromarray(img), return_tensors="pt")
    with torch.no_grad():
        lg = model(**inp).logits  # 1,150,h,w
    lg = torch.nn.functional.interpolate(lg, size=(V, V), mode="bilinear", align_corners=False)
    pr = lg.softmax(1)[0].numpy()
    out = {g: pr[ids].sum(0) for g, ids in GROUPS.items()}
    return out

def run(path, outp, nadir=0.842):
    eq = np.asarray(Image.open(path).convert("RGB").resize((4096, 2048), Image.LANCZOS))
    D = dirs_equirect(OW, OH)
    acc = {g: np.zeros((OH, OW), np.float32) for g in GROUPS}; wsum = np.zeros((OH, OW), np.float32)
    views = [(y, 0, 100) for y in range(0, 360, 45)] + [(y, 45, 100) for y in range(0, 360, 90)] + [(y + 45, -35, 100) for y in range(0, 360, 90)]
    for (yaw, pitch, fov) in views:
        img = render_view(eq, yaw, pitch, fov)
        pr = seg(img)
        f, r, u = view_basis(yaw, pitch); k = math.tan(math.radians(fov) / 2)
        z = D @ f; x = (D @ r) / np.maximum(z, 1e-6) / k; y = (D @ u) / np.maximum(z, 1e-6) / k
        ok = (z > 0.2) & (np.abs(x) < 1) & (np.abs(y) < 1)
        ix = np.clip(((x + 1) / 2 * V).astype(int), 0, V - 1); iy = np.clip(((1 - y) / 2 * V).astype(int), 0, V - 1)
        w = np.where(ok, z ** 4, 0).astype(np.float32)
        for g in GROUPS: acc[g] += w * pr[g][iy, ix]
        wsum += w
        print(" view", yaw, pitch, flush=True)
    for g in GROUPS: acc[g] /= np.maximum(wsum, 1e-6)
    acc["nadir"] = (np.arange(OH)[:, None] / OH >= nadir) * np.ones((1, OW))
    np.savez_compressed(outp, **acc)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])

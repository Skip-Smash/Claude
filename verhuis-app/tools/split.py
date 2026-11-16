import sys, json, numpy as np
from PIL import Image
from scipy.ndimage import median_filter, uniform_filter1d, binary_opening, label as cc_label

def corners_from(d, H=1024, W=2048):
    wall = d['wall'] > .5; ceil = d['ceiling'] > .5
    y0, y1 = int(.12 * H), int(.55 * H)
    yt = np.full(W, np.nan)
    for x in range(W):
        col = wall[y0:y1, x]
        idx = np.where(col)[0]
        if len(idx) == 0: continue
        y = idx[0] + y0
        if ceil[max(0, y - 6):y, x].mean() > .5: yt[x] = y
    # gaten opvullen (circulair)
    xs = np.arange(W); ok = ~np.isnan(yt)
    if ok.sum() < W * .2: return [], yt
    yt_f = np.interp(xs, np.concatenate([xs[ok] - W, xs[ok], xs[ok] + W]), np.concatenate([yt[ok]] * 3))
    ys = median_filter(yt_f, size=21, mode='wrap')
    return ys

def find_corners(ys, W=2048, prom=10, minsep=0.05):
    # hoeken = lokale maxima van de plafondlijn (verste punten)
    n = len(ys); win = int(W * minsep / 2)
    cands = []
    for x in range(n):
        seg = np.take(ys, range(x - win, x + win + 1), mode='wrap')
        if ys[x] >= seg.max() - 1e-6:
            # prominentie: verschil met minima links/rechts binnen 2*win
            l = np.take(ys, range(x - 3 * win, x), mode='wrap').min(); r = np.take(ys, range(x + 1, x + 3 * win + 1), mode='wrap').min()
            if ys[x] - max(l, r) >= prom: cands.append(x)
    # samenvoegen van vlakke toppen
    out = []
    for x in cands:
        if out and (x - out[-1][-1]) <= 3: out[-1].append(x)
        else: out.append([x])
    if len(out) > 1 and (out[0][0] + W - out[-1][-1]) <= 3: out[0] = out[-1] + out[0]; out.pop()
    return [int(np.median(g)) % W for g in out]

def wall_masks(d, corners, W=2048, H=1024):
    p = d['wall'].copy(); p[d['nadir'] > 0] = 0
    alpha = np.clip((p - .35) / .3, 0, 1)
    walls = []
    cs = sorted(corners)
    if len(cs) < 2: cs = [0, W // 2] if not cs else [cs[0], (cs[0] + W // 2) % W]
    for i in range(len(cs)):
        a, b = cs[i], cs[(i + 1) % len(cs)]
        m = np.zeros((H, W), np.float32)
        if b > a: m[:, a:b] = alpha[:, a:b]
        else: m[:, a:] = alpha[:, a:]; m[:, :b] = alpha[:, :b]
        area = (m > .5).sum()
        if area < H * W * .0015: continue
        # alleen grote samenhangende stukken
        lab, n = cc_label(m > .5)
        if n > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            keep = np.isin(lab, np.where(sizes >= max(area * .08, 400))[0])
            m = m * (keep | ((m > 0) & (m <= .5)))
        cu = ((a + ((b - a) % W) / 2) % W) / W
        walls.append({"a": a, "b": b, "center_u": round(cu, 4), "mask": (m * 255).astype(np.uint8), "area": int(area)})
    return walls

def vis(eq_path, walls, ys, corners, out):
    eq = np.asarray(Image.open(eq_path).convert('RGB').resize((2048, 1024)), dtype=np.float32)
    cols = np.array([[230, 40, 160], [40, 120, 240], [240, 150, 20], [30, 200, 90], [160, 60, 230], [230, 220, 30], [20, 200, 210], [240, 80, 60]], np.float32)
    o = eq.copy()
    for i, w in enumerate(walls):
        a = (w['mask'] / 255.)[..., None] * .55
        o = o * (1 - a) + cols[i % len(cols)] * a
    for x in corners: o[:, max(0, x - 1):x + 2] = [255, 255, 0]
    for x in range(0, 2048, 2):
        y = int(ys[x]) if not np.isnan(ys[x]) else None
        if y: o[max(0, y - 1):y + 1, x] = [255, 0, 0]
    im = Image.fromarray(o.astype(np.uint8)).resize((1400, 700))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    for i, w in enumerate(walls): dr.text((w['center_u'] * 1400, 360), str(i + 1), fill=(255, 255, 255))
    im.save(out)

if __name__ == "__main__":
    eq, npz, out = sys.argv[1:4]; extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    d = dict(np.load(npz))
    ys = corners_from(d)
    corners = extra.get("corners") or find_corners(ys, prom=extra.get("prom", 10))
    walls = wall_masks(d, corners)
    vis(eq, walls, ys, corners, out)
    print(json.dumps({"corners": corners, "walls": [{"i": i + 1, "a": w['a'], "b": w['b'], "u": w['center_u'], "area": w['area']} for i, w in enumerate(walls)]}))
    if extra.get("save"):
        import os; os.makedirs(extra["save"], exist_ok=True)
        for i, w in enumerate(walls): Image.fromarray(w['mask']).save(f"{extra['save']}/w{i + 1}.png", optimize=True)

"""Non-proteinaceous densities: cryo-EM map density that the atomic model does not explain.

For one layer of a filament: download the EMDB map linked to the PDB entry, sample it in
a thin slab around the layer, blank out everything within MASK_RADIUS of a modelled atom,
and keep the remaining blobs that sit on or inside the filament. Returns filled contour
polygons (in the same 2D frame as the Cα trace) at two levels, for drawing.
"""
import os
import urllib.request

import contourpy
import gemmi
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

CACHE = os.path.expanduser("~/.cache/amyloid_emdb")
STEP = 0.5            # Å, in-plane sampling
SLAB = 2.4            # Å either side of the layer
MASK_RADIUS = 2.5     # Å around every modelled atom
RISE = 4.75           # Å; atoms are also copied one layer up and down before masking
MARGIN = 12.0         # Å around the layer
MAX_DIST = 8.0        # Å; blobs further than this from the layer's atoms are ignored
RAYS = 24             # directions checked for "inside the core"
RAY_LENGTH = 30.0     # Å
CONTACT = 5.0         # Å; a kept blob must come this close to a modelled atom
MIN_DRAW_AREA = 14.0  # Å²; small densities are drawn at least this big (keeping their shape)


def buriedness(point, tree):
    """Fraction of in-plane directions from point that run into protein within RAY_LENGTH."""
    steps = np.arange(1.0, RAY_LENGTH, 1.0)
    hits = 0
    for a in np.linspace(0, 2 * np.pi, RAYS, endpoint=False):
        ray = point + np.outer(steps, [np.cos(a), np.sin(a)])
        if (tree.query(ray)[0] < 2.5).any():
            hits += 1
    return hits / RAYS


def emdb_id(cif_path):
    """The EMDB map deposited with this model (not maps of related entries)."""
    block = gemmi.cif.read(cif_path).sole_block()
    related = block.find("_pdbx_database_related.", ["db_name", "db_id", "?content_type"])
    emdb = []
    for row in related:
        if gemmi.cif.as_string(row[0]).upper() != "EMDB":
            continue
        kind = gemmi.cif.as_string(row[2]).lower() if row.has(2) else ""
        emdb.append((kind != "associated em volume", gemmi.cif.as_string(row[1]).upper().replace("EMD-", "")))
    return min(emdb)[1] if emdb else None


def fetch_map(num):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"emd_{num}.map.gz")
    if not os.path.exists(path):
        url = f"https://files.wwpdb.org/pub/emdb/structures/EMD-{num}/map/emd_{num}.map.gz"
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def unexplained(cif_path, layer, frame, threshold=0.22, peak=0.35, min_area=5.0, inside=0.0, report=None,
                per_protofilament=True, pick=None):
    """Returns (u, v, image, level): masked density in the layer plane, and the drawing level.

    Both levels are fractions of the median map value at the layer's Cα atoms, so that
    maps with different scaling look alike.
    threshold: outline of the drawn density. Lower = bigger shapes.
    peak:      a blob is only kept if its strongest point reaches this. Lower = more blobs.
               (Keeps out the broad, faint fuzzy-coat density around the filament.)
    min_area:  Å², smaller blobs are dropped.
    inside:    minimum buriedness (0-1): the fraction of directions from the blob that run into
               the filament. Around 0.75 keeps only densities inside the core / between
               protofilaments; 0 keeps everything.
    report:    optional list; (u, v, buriedness, peak) of every candidate blob is appended.
    per_protofilament: keep only the most enclosed (then strongest) density next to each chain.
    pick:      list of candidate numbers to keep instead (numbers as returned in `candidates`).

    Returns (u, v, image, level, candidates), candidates = [(number, u, v, kept)], numbered
    from most to least enclosed.
    """
    num = emdb_id(cif_path)
    if not num:
        raise ValueError("no EMDB map linked to this entry")
    m = gemmi.read_ccp4_map(fetch_map(num))
    m.setup(0.0)
    grid = m.grid

    c, ex, ey = frame
    axis = np.cross(ex, ey)
    st = gemmi.read_structure(cif_path)
    atoms = np.array([a.pos.tolist() for ch in st[0] for r in ch for a in r])
    layer_atoms = np.array([a.pos.tolist() for ch in st[0] if ch.name in layer for r in ch for a in r])
    layer_ca = np.array([a.pos.tolist() for ch in st[0] if ch.name in layer for r in ch for a in r if a.name == "CA"])
    atom_chain = np.array([ch.name for ch in st[0] if ch.name in layer for r in ch for a in r])

    # in-plane box around the layer
    uv = np.c_[(layer_atoms - c) @ ex, (layer_atoms - c) @ ey]
    lo, hi = uv.min(0) - MARGIN, uv.max(0) + MARGIN
    nu, nv = (np.ceil((hi - lo) / STEP)).astype(int) + 1
    ws = np.linspace(-SLAB, SLAB, 7)

    # sample the map on a (nu, nv, nw) block in the layer frame
    arr = np.zeros((nu, nv, len(ws)), dtype=np.float32)
    origin = c + lo[0] * ex + lo[1] * ey + ws[0] * axis
    tr = gemmi.Transform()
    tr.mat.fromlist(np.c_[ex * STEP, ey * STEP, axis * (ws[1] - ws[0])].tolist())
    tr.vec.fromlist(origin.tolist())
    grid.interpolate_values(arr, tr)

    # mask density explained by the model (plus copies one layer up/down)
    pts = (origin + np.stack(np.meshgrid(np.arange(nu) * STEP, np.arange(nv) * STEP,
                                         np.arange(len(ws)) * (ws[1] - ws[0]), indexing="ij"), -1)
           .reshape(-1, 3) @ np.array([ex, ey, axis]))
    near = np.abs((atoms - c) @ axis) < SLAB + MASK_RADIUS + RISE
    masked_atoms = np.vstack([atoms[near] + k * RISE * axis for k in (-1, 0, 1)])
    dist, _ = cKDTree(masked_atoms).query(pts, distance_upper_bound=MASK_RADIUS)
    arr.reshape(-1)[np.isfinite(dist)] = 0.0
    slab = arr.mean(axis=2)
    img = ndimage.gaussian_filter(slab, sigma=0.6 / STEP)       # for finding and judging blobs
    smooth = ndimage.gaussian_filter(slab, sigma=0.9 / STEP)    # for drawing them

    # level relative to the protein density of this map, so different maps look alike
    ca_val = np.array([grid.interpolate_value(gemmi.Position(*p)) for p in layer_ca])
    med = float(np.median(ca_val))
    noise = float(np.array(grid, copy=False).std())
    if med < 5 * noise:
        raise ValueError(f"EMD-{num} does not line up with the model")
    level = threshold * med

    # candidate blobs: big enough, strong enough, touching and enclosed by the filament
    labels, n = ndimage.label(img > level)
    cands = []
    if n:
        areas = ndimage.sum(np.ones_like(img), labels, index=np.arange(1, n + 1)) * STEP ** 2
        tree = cKDTree(uv)
        for i in range(1, n + 1):
            if areas[i - 1] < min_area or img[labels == i].max() < peak * med:
                continue
            ii, jj = np.nonzero(labels == i)
            if ii.min() == 0 or jj.min() == 0 or ii.max() == nu - 1 or jj.max() == nv - 1:
                continue  # touches the edge of the box: outside the filament
            blob_uv = np.c_[lo[0] + ii * STEP, lo[1] + jj * STEP]
            dists, nearest = tree.query(blob_uv)
            if dists.min() > MAX_DIST:
                continue
            k = np.argmax(img[ii, jj])
            top = blob_uv[k]
            b = buriedness(top, tree)
            pk = img[ii[k], jj[k]] / med
            if report is not None:
                report.append((top[0], top[1], b, pk))
            if b >= inside and dists.min() <= CONTACT:
                cands.append(dict(label=i, u=top[0], v=top[1], b=b, peak=pk,
                                  chain=atom_chain[nearest[np.argmin(dists)]]))

    # number candidates from most to least enclosed, then strongest
    cands.sort(key=lambda d: (-round(d["b"], 2), -d["peak"]))
    for num, d in enumerate(cands, 1):
        d["num"] = num
    if pick:
        chosen = [d for d in cands if d["num"] in set(int(p) for p in pick)]
    elif per_protofilament:
        chosen, seen = [], set()
        for d in cands:
            if d["chain"] not in seen:
                seen.add(d["chain"])
                chosen.append(d)
    else:
        chosen = cands

    # draw the smoother map around each chosen blob; small ones are scaled up to a visible size
    out = np.zeros_like(img)
    need = int(MIN_DRAW_AREA / STEP ** 2)
    for d in chosen:
        region = ndimage.binary_dilation(labels == d["label"], iterations=int(3.0 / STEP))
        vals = np.sort(smooth[region])[::-1]
        t = vals[min(need, len(vals) - 1)]
        blob = np.where(region, smooth, 0.0)
        if 0 < t < level:
            blob *= level / t
        out = np.maximum(out, blob)
    img = out
    chosen_nums = {d["num"] for d in chosen}
    candidates = [(d["num"], d["u"], d["v"], d["num"] in chosen_nums) for d in cands]

    u = lo[0] + np.arange(nu) * STEP
    v = lo[1] + np.arange(nv) * STEP
    return u, v, img, level, candidates


def contours(u, v, img, levels):
    """For each level: a list of rings (N x 2 arrays in (u, v)) enclosing density above it."""
    if not img.any():
        return [[] for _ in levels]
    gen = contourpy.contour_generator(v, u, img, fill_type=contourpy.FillType.OuterOffset)
    out = []
    for lv in levels:
        rings = []
        pts_list, offs_list = gen.filled(lv, np.inf)
        for pts, offs in zip(pts_list, offs_list):
            rings += [pts[a:b][:, ::-1] for a, b in zip(offs[:-1], offs[1:])]  # back to (u, v)
        out.append(rings)
    return out

#!/usr/bin/env python3
"""Draw a single cross-sectional layer of an amyloid filament as a Cα trace.

Usage:
    python amyloid_layer.py 5O3L 6NWP my_model.cif -o out/ --style residues
    python amyloid_layer.py --list ids.txt -o out/

For each structure: finds the fibril axis from the chain stacking, picks one
layer (one chain per protofilament) from the middle of the stack, projects
the Cα atoms onto the plane perpendicular to the axis, and writes an SVG.
All structures are drawn at the same Å scale, and a PNG contact sheet of all
of them is written to <out>/montage.png.
"""
import argparse
import math
import os
import urllib.request

import gemmi
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["svg.fonttype"] = "none"  # keep text editable in Illustrator/Inkscape
plt.rcParams["font.family"] = "Helvetica, Arial, DejaVu Sans".split(", ")

# Residue classes as in the usual amyloid fold schematics
RES_CLASS = {
    **dict.fromkeys("AVILMFWC", "hydrophobic"),
    **dict.fromkeys("STNQYH", "polar"),
    **dict.fromkeys("KR", "positive"),
    **dict.fromkeys("DE", "negative"),
    "G": "glycine",
    "P": "proline",
}
CLASS_COLOR = {
    "hydrophobic": "#f2f2f2",
    "polar": "#69b3e7",
    "positive": "#e5484d",
    "negative": "#f5a623",
    "glycine": "#e0d4f5",
    "proline": "#b8e0b0",
    "other": "#cccccc",
}
CHAIN_COLORS = ["#1f4e9c", "#c0392b", "#2a9d5c", "#8e44ad", "#d68910", "#17a2b8"]

LAYER_SPACING = 4.8  # Å
CA_BREAK = 4.3  # Å; longer Cα–Cα distance = chain break


# ----------------------------------------------------------------- loading
def fetch(pdb_id, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id.upper()}.cif")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        urllib.request.urlretrieve(url, path)
    return path


def load_chains(path, min_res=8):
    """Return {chain_name: [(seqnum, one_letter, xyz), ...]} for protein chains (model 1)."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    title = ""
    if path.lower().endswith((".cif", ".mmcif")):
        block = gemmi.cif.read(path).sole_block()
        method = " ".join(block.find_values("_exptl.method")).upper()
        if method and "ELECTRON MICROSCOPY" not in method:
            raise ValueError(f"not a cryo-EM structure ({method.strip()}) - skipped")
        title = gemmi.cif.as_string(block.find_value("_struct.title") or "")
    chains = {}
    for ch in st[0]:
        poly = ch.get_polymer()
        if poly.check_polymer_type() not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.Unknown):
            continue
        cas = []
        for res in poly:
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            aa = gemmi.find_tabulated_residue(res.name)
            code = aa.one_letter_code.upper() if aa and aa.is_amino_acid() else "X"
            cas.append((res.seqid.num, code, np.array(ca.pos.tolist())))
        if len(cas) >= min_res:
            chains[ch.name] = cas
    return chains, title.strip()


def chain_proteins(path):
    """{chain_name: (protein description, full sequence length)} from the mmCIF entity records."""
    if not path.lower().endswith((".cif", ".mmcif")):
        return {}
    block = gemmi.cif.read(path).sole_block()
    desc = dict(zip(block.find_values("_entity.id"), block.find_values("_entity.pdbx_description")))
    seqs = dict(zip(block.find_values("_entity_poly.entity_id"),
                    block.find_values("_entity_poly.pdbx_seq_one_letter_code_can")))
    out = {}
    scheme = block.find("_pdbx_poly_seq_scheme.", ["entity_id", "pdb_strand_id"])
    for row in scheme:
        ent, chain = row[0], row[1]
        if chain not in out:
            seq = gemmi.cif.as_string(seqs.get(ent, "")).replace("\n", "").replace(" ", "")
            out[chain] = (gemmi.cif.as_string(desc.get(ent, "?")), len(seq))
    return out


# ---------------------------------------------------------- fibril geometry
def matched_displacement(a, b):
    """Mean displacement b - a over residues with the same number, and RMS spread."""
    da = {n: x for n, _, x in a}
    common = [n for n, _, _ in b if n in da]
    if len(common) < 5:
        return None, None
    db = {n: x for n, _, x in b}
    d = np.array([db[n] - da[n] for n in common])
    mean = d.mean(axis=0)
    return mean, np.sqrt(((d - mean) ** 2).sum(axis=1).mean())


def find_axis(chains):
    """Fibril axis from displacements between chains stacked on top of each other."""
    names = list(chains)
    vecs = []
    for i, a in enumerate(names):
        best = None
        for b in names:
            if a == b:
                continue
            m, spread = matched_displacement(chains[a], chains[b])
            if m is None:
                continue
            L = np.linalg.norm(m)
            # a stacking neighbour: ~one or half a layer away, rigid (low spread)
            if 2.0 < L < 6.5 and spread < 3.0 and (best is None or L < best[0]):
                best = (L, m)
        if best is not None:
            vecs.append(best[1] / best[0])
    if not vecs:
        # Only one layer deposited: cryo-EM helical models sit in the map box
        # with the fibril axis along z.
        print("    (only one layer in the file - assuming fibril axis along z)")
        return np.array([0.0, 0.0, 1.0])
    vecs = np.array(vecs)
    ref = vecs[0]
    vecs = np.array([v if v @ ref > 0 else -v for v in vecs])
    axis = vecs.mean(axis=0)
    return axis / np.linalg.norm(axis)


def pick_layer(chains, axis):
    """One chain per protofilament, from the middle of the stack."""
    cent = {n: np.mean([x for _, _, x in c], axis=0) for n, c in chains.items()}
    z = {n: cent[n] @ axis for n in chains}
    # in-plane positions of chain centroids
    inplane = {n: cent[n] - z[n] * axis for n in chains}

    # group chains into protofilaments by in-plane centroid position
    pfs = []
    for n in sorted(chains, key=lambda n: z[n]):
        for pf in pfs:
            if np.linalg.norm(inplane[n] - inplane[pf[0]]) < 6.0:
                pf.append(n)
                break
        else:
            pfs.append([n])
    # drop singleton "protofilaments" if the others are real stacks (ligands, antibodies ...)
    if any(len(pf) > 1 for pf in pfs):
        pfs = [pf for pf in pfs if len(pf) > 1]

    zmid = np.median([z[n] for pf in pfs for n in pf])
    ref_pf = max(pfs, key=len)
    ref = min(ref_pf, key=lambda n: abs(z[n] - zmid))
    layer = [min(pf, key=lambda n: abs(z[n] - z[ref])) for pf in pfs]
    return layer, pfs


def projection_frame(chains, layer, axis, flip=False, rotate=0.0):
    """2D basis perpendicular to the axis; long dimension horizontal, centred."""
    pts = np.array([x for n in layer for _, _, x in chains[n]])
    if flip:
        axis = -axis
    c = pts.mean(axis=0)
    p = pts - c
    p = p - np.outer(p @ axis, axis)
    # principal in-plane direction -> x
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    ex = vt[0] - (vt[0] @ axis) * axis
    ex /= np.linalg.norm(ex)
    ey = np.cross(axis, ex)  # right-handed: viewer looks down -axis
    # deterministic sign: put the first chain's N-terminus on the left
    first = chains[layer[0]][0][2] - c
    if first @ ex > 0:
        ex, ey = -ex, -ey
    th = math.radians(rotate)
    ex, ey = math.cos(th) * ex + math.sin(th) * ey, -math.sin(th) * ex + math.cos(th) * ey
    return c, ex, ey


def project(chains, layer, frame):
    c, ex, ey = frame
    out = {}
    for n in layer:
        out[n] = [(num, aa, np.array([(x - c) @ ex, (x - c) @ ey])) for num, aa, x in chains[n]]
    return out


# ----------------------------------------------------------------- drawing
def segments(res, chains3d):
    """Split a chain into continuous pieces at Cα–Cα breaks (measured in 3D)."""
    segs, cur = [], [0]
    for i in range(1, len(res)):
        if np.linalg.norm(chains3d[i][2] - chains3d[i - 1][2]) > CA_BREAK:
            segs.append(cur)
            cur = []
        cur.append(i)
    segs.append(cur)
    return segs


def draw(ax, proj, chains, style="residues", label_every=10, title=None, half=None):
    for k, (n, res) in enumerate(proj.items()):
        col = CHAIN_COLORS[k % len(CHAIN_COLORS)]
        xy = np.array([p for _, _, p in res])
        for seg in segments(res, chains[n]):
            ax.plot(xy[seg, 0], xy[seg, 1], "-", color=col, lw=1.6 if style == "trace" else 1.2,
                    solid_capstyle="round", zorder=1)
        if style == "residues":
            for num, aa, p in res:
                ax.scatter(*p, s=26, c=CLASS_COLOR[RES_CLASS.get(aa, "other")],
                           edgecolors="#333", linewidths=0.5, zorder=2)
        # residue numbers, placed outward from the local chain direction
        for i, (num, aa, p) in enumerate(res):
            is_end = i in (0, len(res) - 1)
            if not (is_end or (label_every and num % label_every == 0)):
                continue
            a, b = xy[max(i - 1, 0)], xy[min(i + 1, len(xy) - 1)]
            t = b - a
            nrm = np.array([-t[1], t[0]]) / (np.linalg.norm(t) + 1e-9)
            q = p + nrm * 2.6
            lbl = f"{'N' if i == 0 else 'C'}{num}" if is_end else str(num)
            ax.text(*q, lbl, fontsize=5.5, ha="center", va="center", color=col, zorder=3,
                    fontweight="bold" if is_end else "normal")
    ax.set_aspect("equal")
    ax.axis("off")
    if half is not None:
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        # 10 Å scale bar, bottom-left
        x0, y0 = -half + 3, -half + 3
        ax.plot([x0, x0 + 10], [y0, y0], "k-", lw=1.5)
        ax.text(x0 + 5, y0 + 1.5, "10 Å", fontsize=6, ha="center", va="bottom")
    if title:
        ax.set_title(title, fontsize=8, pad=2)


# ----------------------------------------------------------------- main
def process(src, args):
    path = fetch(src, args.cache) if not os.path.exists(src) else src
    name = os.path.splitext(os.path.basename(src))[0].upper()
    chains, title = load_chains(path)
    axis = find_axis(chains)
    layer, pfs = pick_layer(chains, axis)
    frame = projection_frame(chains, layer, axis, flip=name in args.flip, rotate=0.0)
    proj = project(chains, layer, frame)
    extent = max(np.abs(p).max() for res in proj.values() for _, _, p in res)
    return dict(name=name, title=title, chains=chains, layer=layer, pfs=pfs, proj=proj, extent=extent)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="PDB IDs or local .cif/.pdb files")
    ap.add_argument("--list", help="text file with one PDB ID / path per line")
    ap.add_argument("-o", "--out", default="amyloid_svgs")
    ap.add_argument("--style", choices=["residues", "trace"], default="residues")
    ap.add_argument("--label-every", type=int, default=10, help="label every Nth residue (0 = ends only)")
    ap.add_argument("--flip", nargs="*", default=[], help="IDs to view from the other end of the fibril")
    ap.add_argument("--box", type=float, default=0, help="half-width of drawing in Å (0 = fit largest)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/amyloid_pdb"))
    args = ap.parse_args()
    args.flip = [f.upper() for f in args.flip]

    srcs = list(args.inputs)
    if args.list:
        srcs += [l.split()[0] for l in open(args.list) if l.strip() and not l.startswith("#")]
    os.makedirs(args.out, exist_ok=True)

    results = []
    for s in srcs:
        try:
            r = process(s, args)
        except Exception as e:  # keep going on bad entries
            print(f"{s}: FAILED - {e}")
            continue
        results.append(r)
        print(f"{r['name']}: {len(r['chains'])} chains, {len(r['pfs'])} protofilament(s), "
              f"layer = {'+'.join(r['layer'])}")

    if not results:
        return
    half = args.box or max(r["extent"] for r in results) + 6  # same scale for all

    for r in results:
        fig, ax = plt.subplots(figsize=(half / 12, half / 12))  # 6 Å per cm-ish, constant scale
        draw(ax, r["proj"], r["chains"], args.style, args.label_every, r["name"], half)
        fig.savefig(os.path.join(args.out, f"{r['name']}.svg"), bbox_inches="tight")
        plt.close(fig)

    cols = min(args.cols, len(results))
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, results):
        draw(ax, r["proj"], r["chains"], args.style, args.label_every, r["name"], half)
    if args.style == "residues":
        handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="#333", label=k)
                   for k, c in CLASS_COLOR.items() if k != "other"]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(args.out, "montage.png"), dpi=200)
    fig.savefig(os.path.join(args.out, "montage.svg"))
    print(f"wrote {len(results)} SVGs + montage.png to {args.out}")


if __name__ == "__main__":
    main()

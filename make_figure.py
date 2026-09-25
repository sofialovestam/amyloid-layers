#!/usr/bin/env python3
"""Build the amyloid overview figure described in figure.yaml.

Writes:
    output/figure.svg          editable figure (each structure is a named group)
    output/figure.png          preview
    output/structures/ID.svg   each structure on its own, same scale and style
"""
import math
import os
import sys
from xml.sax.saxutils import escape

import numpy as np
import resvg_py
import yaml

import densities

from amyloid_layer import chain_proteins, fetch, find_axis, load_chains, pick_layer, projection_frame, segments

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/.cache/amyloid_pdb")
DOT_COLOR = "#2bb5d9"
DENSITY_LEVELS = (1.0, 1.4, 2.0)      # nested contours, as multiples of the outline level
DENSITY_SHADES = (0.38, 0.12, 0.0)     # how much each is lightened towards white

MARGIN = 6
COL_GAP = 12
TITLE_H = 13        # family name above its boxes
GROUP_STRIP = 13    # space left of boxes for "3R", "4R" ...
GROUP_PAD = 3       # frame around a group of boxes
BOX_PAD = 6
BOX_GAP = 4         # between boxes of the same group
FAMILY_GAP = 14
STRUCT_GAP = 10
PANEL_LETTER_H = 14
PANEL_GAP = 18

# panel b (core-forming residues)
CORE_LABEL_W = 78   # room for the disease label right of the bars
CORE_NUM_W = 14     # room for the start/end residue numbers
CORE_ROW = 8
CORE_BAR = 2.6


# ------------------------------------------------------------- structures
def layer_traces(spec, page, fam=None):
    """Cα polylines (in Å, centred, x right / y up) for one structure entry."""
    src = str(spec["id"])
    local = os.path.join(HERE, src)
    path = local if os.path.exists(local) else src if os.path.exists(src) else fetch(src, CACHE)
    chains, title = load_chains(path)
    axis = find_axis(chains)
    if spec.get("chains"):
        layer = [str(c) for c in spec["chains"]]
        missing = [c for c in layer if c not in chains]
        if missing:
            raise ValueError(f"chain(s) {missing} not found; available: {', '.join(chains)}")
    else:
        layer, _ = pick_layer(chains, axis)
    c, ex, ey = projection_frame(chains, layer, axis)
    proteins = chain_proteins(path)

    th = math.radians(float(spec.get("rotate", 0)))
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    lines, residues, lengths = [], {}, {}
    for n in layer:
        res = chains[n]
        desc, seqlen = proteins.get(n, ("?", 0))
        residues.setdefault(desc, set()).update(num for num, _, _ in res)
        lengths[desc] = max(lengths.get(desc, 0), seqlen)
        xy = np.array([[(x - c) @ ex, (x - c) @ ey] for _, _, x in res]) @ rot.T
        if spec.get("flip"):
            xy[:, 0] *= -1
        lines += [(xy[seg], desc) for seg in segments(res, res) if len(seg) > 1]
    allxy = np.vstack([l for l, _ in lines])
    mid = (allxy.min(0) + allxy.max(0)) / 2
    lines = [(l - mid, d) for l, d in lines]
    allxy = allxy - mid

    # map density not explained by the model, in the same 2D frame
    blobs, labels, note = [], [], ""
    opt = lambda key, default: spec.get(key, (fam or {}).get(key, page.get(key, default)))
    if opt("densities", True):
        try:
            u, v, img, level, cands = densities.unexplained(
                path, layer, (c, ex, ey),
                threshold=float(opt("density_threshold", 0.15)),
                peak=float(opt("density_peak", 0.2)),
                inside=float(opt("density_inside", 0.72)),
                per_protofilament=opt("density_mode", "per_protofilament") == "per_protofilament",
                pick=spec.get("density_pick"))
            if spec.get("density_numbers"):
                for num, cu, cv, kept in cands:
                    xy = np.array([cu, cv]) @ rot.T
                    if spec.get("flip"):
                        xy[0] *= -1
                    labels.append((num, xy - mid, kept))
            for rings in densities.contours(u, v, img, [level * f for f in DENSITY_LEVELS]):
                out = []
                for r in rings:
                    xy = r @ rot.T
                    if spec.get("flip"):
                        xy[:, 0] *= -1
                    out.append(xy - mid)
                blobs.append(out)
            n = sum(k for *_, k in cands)
            note = f"{n} of {len(cands)} densit{'y' if len(cands) == 1 else 'ies'}"
            if blobs and blobs[0]:
                allxy = np.vstack([allxy] + blobs[0])
        except Exception as e:
            note = f"no densities ({e})"
    return dict(lines=lines, lo=allxy.min(0), hi=allxy.max(0), title=title, layer=layer,
                residues=residues, lengths=lengths, blobs=blobs, note=note, density_labels=labels)


class Proteins:
    """Decides which protein a chain is (family protein, an extra protein, or neither)."""

    def __init__(self, cfg, fam):
        self.fam = fam
        self.extras = cfg.get("extra_proteins") or []

    def extra(self, desc):
        for ex in self.extras:
            if str(ex["match"]).lower() in desc.lower():
                return ex
        return None

    def key(self, desc):
        """'main', an extra-protein name, or None (e.g. unknown fragments)."""
        ex = self.extra(desc)
        if ex:
            return ex["name"]
        match = self.fam.get("protein")
        if match:
            return "main" if str(match).lower() in desc.lower() else None
        return None if "unknown" in desc.lower() else "main"

    def color(self, desc):
        ex = self.extra(desc)
        return ex["color"] if ex else self.fam["color"]


def darken(hex_color, amount):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(v * (1 - amount)):02x}" for v in (r, g, b))


def lighten(hex_color, amount):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda v: round(v + (255 - v) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def structure_svg(s, spec, prot, page, cx, cy):
    """SVG group for one structure centred on (cx, cy) px."""
    k, lw = page["scale"], page["line_width"]
    sid = escape(str(spec["id"]))
    out = [f'<g id="{sid}" data-chains="{"+".join(s["layer"])}">',
           f"<title>{sid} {escape(s['title'])}</title>"]
    if spec.get("grid"):
        out.append(grid_svg(s, k, cx, cy))

    def poly(xy, stroke, width):
        pts = " ".join(f"{cx + x * k:.2f},{cy - y * k:.2f}" for x, y in xy)
        return (f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width:.2f}" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')

    out += density_svg(s, page, cx, cy)
    for xy, desc in s["lines"]:
        out.append(poly(xy, prot.color(desc), lw))
    if page.get("style", "outline") == "outline":
        for xy, desc in s["lines"]:
            out.append(poly(xy, lighten(prot.color(desc), 0.8), lw * 0.4))
    for x, y in spec.get("dots") or []:
        out.append(f'<circle cx="{cx + x * k:.2f}" cy="{cy - y * k:.2f}" r="{lw * 1.5:.2f}" fill="{DOT_COLOR}"/>')
    out.append("</g>")
    return "\n".join(out)


def density_svg(s, page, cx, cy):
    """Nested filled contours: pale outline of the density, stronger shades towards its core."""
    k = page["scale"]
    color = page.get("density_color", DOT_COLOR)
    out = []
    for i, rings in enumerate(s.get("blobs") or []):
        if not rings:
            continue
        d = " ".join("M" + " L".join(f"{cx + x * k:.2f},{cy - y * k:.2f}" for x, y in r) + " Z" for r in rings)
        edge = f' stroke="{darken(color, 0.25)}" stroke-width="0.25"' if i == 0 else ""
        out.append(f'<path class="density" d="{d}" fill="{lighten(color, DENSITY_SHADES[i])}" '
                   f'fill-rule="evenodd"{edge}/>')
    for num, (x, y), kept in s.get("density_labels") or []:
        out.append(f'<text x="{cx + x * k + 1.5:.1f}" y="{cy - y * k - 1.5:.1f}" font-size="4" font-family="Helvetica" '
                   f'fill="{"#d00" if kept else "#888"}">{num}</text>')
    return out


def grid_svg(s, k, cx, cy):
    lo = np.floor(s["lo"] / 10) * 10
    hi = np.ceil(s["hi"] / 10) * 10
    g = ['<g class="grid" stroke="#bbb" stroke-width="0.25" font-size="3" fill="#888">']
    for x in np.arange(lo[0], hi[0] + 1, 10):
        g.append(f'<line x1="{cx + x * k:.1f}" y1="{cy - lo[1] * k:.1f}" x2="{cx + x * k:.1f}" y2="{cy - hi[1] * k:.1f}"/>')
        g.append(f'<text x="{cx + x * k:.1f}" y="{cy - lo[1] * k + 3.5:.1f}" stroke="none" text-anchor="middle">{x:.0f}</text>')
    for y in np.arange(lo[1], hi[1] + 1, 10):
        g.append(f'<line x1="{cx + lo[0] * k:.1f}" y1="{cy - y * k:.1f}" x2="{cx + hi[0] * k:.1f}" y2="{cy - y * k:.1f}"/>')
        g.append(f'<text x="{cx + lo[0] * k - 1:.1f}" y="{cy - y * k + 1:.1f}" stroke="none" text-anchor="end">{y:.0f}</text>')
    g.append("</g>")
    return "\n".join(g)


# ------------------------------------------------------------------ layout
def text(x, y, s, page, anchor="start", size=None, rotate=None, color="#222", bold=False):
    size = size or page["font_size"]
    tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    weight = ' font-weight="bold"' if bold else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{page["font"]}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}"{weight}{tr}>{escape(str(s))}</text>')


def layout_box(box, structs, page, width):
    """Place structures in rows. Returns (height, [(spec, s, cx, cy) relative to box])."""
    k = page["scale"]
    inner = width - 2 * BOX_PAD
    rows, row, row_w = [], [], 0
    for spec, s in structs:
        w = (s["hi"][0] - s["lo"][0]) * k
        if row and row_w + STRUCT_GAP + w > inner:
            rows.append(row)
            row, row_w = [], 0
        row_w += (STRUCT_GAP if row else 0) + w
        row.append((spec, s, w))
    if row:
        rows.append(row)

    placed, y = [], BOX_PAD
    for row in rows:
        h = max((s["hi"][1] - s["lo"][1]) * k if s else 12 for _, s, _ in row)
        free = inner - sum(w for *_, w in row)
        gap = free / (len(row) + 1)   # spread structures evenly across the box
        x = BOX_PAD + gap
        for spec, s, w in row:
            placed.append((spec, s, x + w / 2, y + h / 2))
            x += w + gap
        y += h + STRUCT_GAP
    height = y - STRUCT_GAP + BOX_PAD + page["font_size"] + 2
    return height, placed


def build(cfg):
    page = dict(width=560, scale=0.72, line_width=1.3, style="outline",
                font="Helvetica, Arial, sans-serif", font_size=7)
    page.update(cfg.get("page") or {})
    ncol = int(cfg.get("columns", 3))
    colw = (page["width"] - 2 * MARGIN - (ncol - 1) * COL_GAP) / ncol

    # compute all structures first
    cache, failures = {}, []
    for fam in cfg["families"]:
        for box in fam["boxes"]:
            for spec in box.get("structures") or []:
                key = yaml.safe_dump(spec, sort_keys=True)
                if key in cache:
                    continue
                try:
                    s = layer_traces(spec, page, fam)
                    print(f"  {spec['id']:<8} chains {'+'.join(s['layer']):<8} {s['note']:<18} {s['title'][:60]}")
                except Exception as e:
                    s = None
                    failures.append(f"{spec['id']}: {e}")
                    print(f"  {spec['id']:<8} PROBLEM: {e}")
                cache[key] = s

    body, singles = [], {}
    body.append(text(MARGIN, MARGIN + 9, "a", page, size=page["font_size"] + 4, bold=True))
    col_y = [MARGIN + PANEL_LETTER_H] * ncol
    for fam in cfg["families"]:
        ci = int(fam.get("column", 1)) - 1
        x0 = MARGIN + ci * (colw + COL_GAP)
        y = col_y[ci]
        color = fam["color"]
        prot = Proteins(cfg, fam)
        fam_id = escape(fam["name"]).replace(" ", "_")
        body.append(f'<g id="family_{fam_id}">')
        body.append(text(x0, y + page["font_size"] + 2, fam["name"], page, size=page["font_size"] + 2))
        y += TITLE_H

        has_groups = any(b.get("group") for b in fam["boxes"])
        bx = x0 + (GROUP_STRIP + GROUP_PAD if has_groups else 0)
        bw = colw - (GROUP_STRIP + 2 * GROUP_PAD if has_groups else 0)

        prev_group, group_start = object(), None
        boxes = fam["boxes"]
        for i, box in enumerate(boxes):
            grp = box.get("group")
            if grp != prev_group:
                if i > 0:
                    y += BOX_GAP * 2 if has_groups else BOX_GAP
                group_start = y
                y += GROUP_PAD if grp else 0
            structs = [(spec, cache[yaml.safe_dump(spec, sort_keys=True)]) for spec in box.get("structures") or []]
            ok = [(spec, s) for spec, s in structs if s]
            h, placed = layout_box(box, ok, page, bw)
            body.append(f'<g class="box">')
            body.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="none" '
                        f'stroke="{color}" stroke-width="0.6"/>')
            for spec, s, cx, cy in placed:
                body.append(structure_svg(s, spec, prot, page, bx + cx, y + cy))
                singles[str(spec["id"])] = (spec, s, prot)
            for spec, s in structs:
                if s is None:
                    body.append(text(bx + BOX_PAD, y + BOX_PAD + 6, f"{spec['id']}: see terminal", page, color="#d00"))
            body.append(text(bx + bw - BOX_PAD / 2, y + h - BOX_PAD / 2, box.get("label", ""), page, anchor="end"))
            body.append("</g>")
            y += h

            last_of_group = i == len(boxes) - 1 or boxes[i + 1].get("group") != grp
            if grp and last_of_group:
                y += GROUP_PAD
                body.append(f'<rect x="{bx - GROUP_PAD:.1f}" y="{group_start:.1f}" width="{bw + 2 * GROUP_PAD:.1f}" '
                            f'height="{y - group_start:.1f}" fill="none" stroke="{color}" stroke-width="0.8"/>')
                mid = (group_start + y) / 2
                body.append(text(x0 + GROUP_STRIP / 2 + 2, mid, grp, page, anchor="middle", rotate=-90))
            if not last_of_group:
                y += BOX_GAP
            prev_group = grp
        body.append("</g>")
        col_y[ci] = y + FAMILY_GAP

    y_b = max(col_y) - FAMILY_GAP + PANEL_GAP
    if page.get("densities", True):
        body.append(legend_svg(page, MARGIN, y_b - PANEL_GAP + 10))
        y_b += 8
    if cfg.get("core_residues", True):
        body.append(text(MARGIN, y_b + 9, "b", page, size=page["font_size"] + 4, bold=True))
        y_end = core_panel(cfg, page, cache, body, y_b + PANEL_LETTER_H, colw, ncol)
    else:
        y_end = y_b - PANEL_GAP
    H = y_end + MARGIN
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{page["width"]}" height="{H:.0f}" '
           f'viewBox="0 0 {page["width"]} {H:.0f}">\n<rect width="100%" height="100%" fill="white"/>\n'
           + "\n".join(body) + "\n</svg>\n")
    return svg, singles, page, failures


def legend_svg(page, x, y):
    color = page.get("density_color", DOT_COLOR)
    g = ['<g id="legend">']
    for r, shade in zip((3.2, 2.1, 1.0), DENSITY_SHADES):
        g.append(f'<circle cx="{x + 4:.1f}" cy="{y:.1f}" r="{r}" fill="{lighten(color, shade)}"/>')
    g.append(text(x + 10, y + 2.3, page.get("density_label", "Non-proteinaceous density"), page))
    g.append("</g>")
    return "\n".join(g)


def runs(nums, tolerance=2):
    """Contiguous stretches of residue numbers; gaps of <= tolerance residues are bridged."""
    nums = sorted(nums)
    out = [[nums[0], nums[0]]]
    for n in nums[1:]:
        if n - out[-1][1] <= tolerance + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def core_panel(cfg, page, cache, body, y0, colw, ncol):
    """Panel b: for every protein, its full length and the residues forming each filament core."""
    fs = page["font_size"] - 1
    col_y = [y0] * ncol
    for fam in cfg["families"]:
        prot = Proteins(cfg, fam)
        ci = int(fam.get("column", 1)) - 1
        x0 = MARGIN + ci * (colw + COL_GAP)

        # collect rows: one block per protein, one row per box
        blocks = {}
        for box in fam["boxes"]:
            per = {}
            for spec in box.get("structures") or []:
                s = cache.get(yaml.safe_dump(spec, sort_keys=True))
                if not s:
                    continue
                for desc, nums in s["residues"].items():
                    k = prot.key(desc)
                    if k is None:
                        continue
                    per.setdefault(k, set()).update(nums)
                    b = blocks.setdefault(k, dict(rows=[], seqlen=0, desc=desc))
                    b["seqlen"] = max(b["seqlen"], s["lengths"].get(desc, 0))
            for k, nums in per.items():
                blocks[k]["rows"].append((box.get("core_label", box.get("label", "")), nums))

        y = col_y[ci]
        order = ["main"] + [k for k in blocks if k != "main"]
        for k in [k for k in order if k in blocks]:
            b = blocks[k]
            if k == "main":
                name, color = fam.get("core_name", fam["name"]), fam["color"]
                length = fam.get("length") or b["seqlen"]
            else:
                ex = prot.extra(b["desc"])
                name, color, length = ex["name"], ex["color"], ex.get("length") or b["seqlen"]
            length = max(length, max(max(n) for _, n in b["rows"]))
            first = min(min(n) for _, n in b["rows"])

            bx0 = x0 + CORE_NUM_W
            bx1 = x0 + colw - CORE_LABEL_W - CORE_NUM_W
            broken = first > 0.3 * length
            start = max(1, first - round(0.06 * length)) if broken else 1
            lead = 8 if broken else 0

            def X(r):
                return bx0 + lead + (r - start) / max(1, length - start) * (bx1 - bx0 - lead)

            body.append(f'<g class="core" id="core_{escape(name).replace(" ", "_")}">')
            body.append(text(x0, y + fs + 2, name, page, size=fs + 1.5))
            y += fs + 7
            # full-length line
            gl = "#555"
            if broken:
                body.append(f'<line x1="{bx0:.1f}" y1="{y:.1f}" x2="{bx0 + 3:.1f}" y2="{y:.1f}" stroke="{gl}" stroke-width="0.6"/>')
                for dx in (3.6, 5.2):
                    body.append(f'<line x1="{bx0 + dx - 0.8:.1f}" y1="{y + 2:.1f}" x2="{bx0 + dx + 0.8:.1f}" y2="{y - 2:.1f}" stroke="{gl}" stroke-width="0.5"/>')
            body.append(f'<line x1="{X(start):.1f}" y1="{y:.1f}" x2="{X(length):.1f}" y2="{y:.1f}" stroke="{gl}" stroke-width="0.6"/>')
            body.append(text(bx0 - 2, y + fs * 0.35, 1, page, anchor="end", size=fs))
            body.append(text(X(length) + 2, y + fs * 0.35, length, page, size=fs))
            y += CORE_ROW
            for label, nums in b["rows"]:
                rr = runs(nums)
                for i, (a, z) in enumerate(rr):
                    body.append(f'<line x1="{X(a):.1f}" y1="{y:.1f}" x2="{X(z):.1f}" y2="{y:.1f}" stroke="{color}" '
                                f'stroke-width="{CORE_BAR}"/>')
                    if i:
                        body.append(f'<line x1="{X(rr[i - 1][1]):.1f}" y1="{y:.1f}" x2="{X(a):.1f}" y2="{y:.1f}" '
                                    f'stroke="{color}" stroke-width="0.6" stroke-dasharray="1 1"/>')
                body.append(text(X(rr[0][0]) - 2, y + fs * 0.35, rr[0][0], page, anchor="end", size=fs))
                body.append(text(X(rr[-1][1]) + 2, y + fs * 0.35, rr[-1][1], page, size=fs))
                body.append(text(x0 + colw, y + fs * 0.35, label, page, anchor="end", size=fs))
                y += CORE_ROW
            body.append("</g>")
            y += 8
        col_y[ci] = y + 4
    return max(col_y)


def single_svg(spec, s, prot, page):
    k = page["scale"]
    w = (s["hi"][0] - s["lo"][0]) * k + 8
    h = (s["hi"][1] - s["lo"][1]) * k + 8
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}" height="{h:.1f}" viewBox="0 0 {w:.1f} {h:.1f}">\n'
            + structure_svg(s, spec, prot, page, w / 2, h / 2) + "\n</svg>\n")


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figure.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    out = os.path.join(HERE, "output")
    os.makedirs(os.path.join(out, "structures"), exist_ok=True)

    print("Structures:")
    svg, singles, page, failures = build(cfg)
    with open(os.path.join(out, "figure.svg"), "w") as f:
        f.write(svg)
    png = resvg_py.svg_to_bytes(svg_string=svg, zoom=4, background="white")
    with open(os.path.join(out, "figure.png"), "wb") as f:
        f.write(bytes(png))
    for sid, (spec, s, prot) in singles.items():
        with open(os.path.join(out, "structures", f"{os.path.basename(sid).split('.')[0]}.svg"), "w") as f:
            f.write(single_svg(spec, s, prot, page))

    print(f"\nDone: output/figure.svg, output/figure.png and {len(singles)} files in output/structures/")
    if failures:
        print("\nThese need attention:")
        for msg in failures:
            print("  -", msg)


if __name__ == "__main__":
    main()

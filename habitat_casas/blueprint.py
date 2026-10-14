"""Floor-plan "blueprint" of the apartment with every sensor marked.

    python -m habitat_casas.blueprint

Writes outputs/blueprint.png (plan + legend, for printing/reading) and
outputs/blueprint_plan.png (plan only, for slides). The plan is turned
sideways (the apartment is long and narrow) so it fits a landscape page:
left = back of the apartment (front door), right = dining/TV corner,
top = kitchen wall.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from habitat_casas.env import REPO_ROOT
from habitat_casas.replay import RES, build_map
from habitat_casas.scenario_apt import CONTACTS, ITEMS, MOTION

OUT_DIR = os.path.join(REPO_ROOT, "outputs")

# plain-language names for every sensor
NAMES = {
    "M001": "Entry, by the front door", "M002": "Desk corner", "M003": "Kitchen: sink end",
    "M004": "Kitchen: cooking spot", "M005": "Kitchen: fridge end", "M006": "Living room: dresser",
    "M007": "Sofa", "M008": "Hallway", "M009": "Dining table", "M010": "TV corner",
    "M011": "Medicine cabinet", "M012": "East room (bike)", "M013": "Whole kitchen",
    "M014": "Whole living room", "M015": "Whole dining + TV area",
    "D001": "Front door", "D002": "Fridge, top door", "D003": "Fridge, bottom door",
    "D004": "Kitchen drawer 1", "D005": "Medicine cabinet door", "D006": "Dresser drawer",
    "D007": "Kitchen cupboard door", "D008": "Kitchen drawer 2", "D009": "Kitchen drawer 3",
    "I001": "Plum (in fridge)", "I002": "Bowl (in fridge)", "I003": "Coffee can (counter)",
    "I004": "Strawberry (counter)", "I005": "Bowl (counter)",
}
AREA_SENSORS = {"M013", "M014", "M015"}

# furniture labels (x, z) from the scene file
FURNITURE = {
    "Fridge": (-1.95, 3.75), "Kitchen counter": (-1.95, 2.1), "Sofa": (1.51, 2.81),
    "Dining table": (0.81, 6.0), "TV": (0.41, 6.09), "Desk": (-2.05, -1.17),
    "Dresser": (-0.02, 0.62), "Medicine\ncabinet": (3.45, 5.75), "Wardrobe": (0.46, 4.63),
    "Beanbag": (2.38, 7.4), "Bike": (3.35, 0.1), "Plant": (0.02, -0.19),
}
ROOMS = {  # label position (x, z)
    "KITCHEN": (-1.25, 2.2), "LIVING ROOM": (2.3, 1.2), "DINING & TV": (2.0, 6.0),
    "ENTRY": (-0.4, -1.7), "BACK ROOM\n(used as 'outside')": (-1.2, -3.9),
}

BG, FLOOR, SOLID = "#1B3A6B", "#2C5A9A", "#13294D"
LINE, ZONE, AREA, DOOR, ITEM = "#E8F1FF", "#8FD3FF", "#FFD166", "#7CE38B", "#FF8FA3"


def draw_plan(ax, amap):
    g = amap["grid"]
    x0, z0 = amap["x0"], amap["z0"]
    nz, nx = g.shape
    # rotated: horizontal axis = z, vertical axis = x (kitchen at the top)
    img = np.where(g.T, 1.0, 0.0)  # rows = x, cols = z
    cmap = matplotlib.colors.ListedColormap([SOLID, FLOOR])
    ax.imshow(img, cmap=cmap, extent=[z0, z0 + nz * RES, x0 + nx * RES, x0],
              interpolation="nearest", zorder=0)
    ax.set_facecolor(BG)

    for label, (x, z) in ROOMS.items():
        ax.text(z, x, label, color=LINE, alpha=0.35, fontsize=15, fontweight="bold",
                ha="center", va="center", zorder=1)
    for label, (x, z) in FURNITURE.items():
        ax.text(z, x, label, color=LINE, alpha=0.75, fontsize=8.5, style="italic",
                ha="center", va="center", zorder=2)

    for m in MOTION:
        area = m.sensor_id in AREA_SENSORS
        r = patches.Rectangle((m.z[0], m.x[0]), m.z[1] - m.z[0], m.x[1] - m.x[0],
                              fill=False, lw=2.2 if area else 1.6, ls=(0, (5, 4)) if area else "-",
                              ec=AREA if area else ZONE, zorder=3)
        ax.add_patch(r)
        if area:
            ax.text(m.z[1] - 0.06, m.x[1] - 0.08, m.sensor_id, color=AREA, fontsize=10,
                    fontweight="bold", ha="right", va="bottom", zorder=5)
        else:
            ax.text(m.z[0] + 0.06, m.x[0] + 0.06, m.sensor_id, color=ZONE, fontsize=10,
                    fontweight="bold", ha="left", va="top", zorder=5)

    # markers at their true spots; labels next to them, except along the
    # crowded kitchen wall, where labels go in a row above the plan
    marks = []
    for c in CONTACTS:
        x, z = amap["contacts"][c.sensor_id]
        if c.sensor_id == "D009":  # stacked under D008: same floor spot
            z += 0.2
        marks.append((c.sensor_id, x, z, DOOR, "s"))
    for it in ITEMS:
        x, z = amap["items"][it.sensor_id]
        marks.append((it.sensor_id, x, z, ITEM, "o"))
    top = amap["x0"] - 0.55  # label row above the kitchen wall
    wall = sorted([m for m in marks if m[1] < -1.8], key=lambda m: m[2])
    slots, last = [], -1e9
    for m in wall:
        z = max(m[2], last + 0.55)
        slots.append(z)
        last = z
    shift = (slots[-1] - wall[-1][2]) / 2 if wall else 0  # recentre the row
    label_at = {m[0]: (top, z - shift) for m, z in zip(wall, slots)}
    for sid, x, z, color, shape in marks:
        if shape == "s":
            ax.add_patch(patches.Rectangle((z - 0.09, x - 0.09), 0.18, 0.18, fc=color, ec=BG, lw=1, zorder=6))
        else:
            ax.add_patch(patches.Circle((z, x), 0.08, fc=color, ec=BG, lw=1, zorder=6))
        if sid in label_at:
            lx, lz = label_at[sid]
            ax.plot([z, lz], [x - 0.1, lx + 0.12], color=color, lw=0.8, alpha=0.8, zorder=5)
            ax.text(lz, lx, sid, color=color, fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=6)
        else:
            ax.text(z, x + 0.16, sid, color=color, fontsize=9.5, fontweight="bold", ha="center", va="top", zorder=6)

    ax.set_xlim(z0 - 0.1, z0 + nz * RES + 0.1)
    ax.set_ylim(x0 + nx * RES + 0.1, x0 - 0.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)
    # scale bar: 1 m
    zs, xs = z0 + 0.3, x0 + nx * RES - 0.25
    ax.plot([zs, zs + 1], [xs, xs], color=LINE, lw=3, zorder=7)
    ax.text(zs + 0.5, xs - 0.1, "1 m", color=LINE, fontsize=10, ha="center", va="bottom", zorder=7)


def legend_handles():
    return [
        patches.Patch(fc=FLOOR, ec=LINE, label="Floor you can walk on"),
        patches.Patch(fc=SOLID, ec=LINE, label="Walls and furniture"),
        plt.Line2D([], [], color=ZONE, lw=2, label="Motion sensor: one area"),
        plt.Line2D([], [], color=AREA, lw=2, ls=(0, (5, 4)), label="Motion sensor: whole room"),
        plt.Line2D([], [], marker="s", ls="", mfc=DOOR, mec=BG, ms=11, label="Door / drawer sensor"),
        plt.Line2D([], [], marker="o", ls="", mfc=ITEM, mec=BG, ms=10, label="Item sensor"),
    ]


def main():
    amap = build_map()
    os.makedirs(OUT_DIR, exist_ok=True)

    # plan only (slides)
    fig, ax = plt.subplots(figsize=(16, 9.0), facecolor=BG)
    draw_plan(ax, amap)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, fontsize=10.5,
               facecolor=BG, edgecolor=LINE, labelcolor=LINE, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.02))
    p1 = os.path.join(OUT_DIR, "blueprint_plan.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # plan + full legend (printing)
    fig = plt.figure(figsize=(16, 15), facecolor=BG)
    ax = fig.add_axes([0.02, 0.38, 0.96, 0.57])
    draw_plan(ax, amap)
    fig.legend(handles=legend_handles(), loc="center", ncol=6, fontsize=10.5,
               facecolor=BG, edgecolor=LINE, labelcolor=LINE, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.375))
    fig.text(0.02, 0.985, "Apartment blueprint: where every sensor is", color=LINE,
             fontsize=20, fontweight="bold", va="top")
    groups = [("Motion sensors", ZONE, [m.sensor_id for m in MOTION]),
              ("Door & drawer sensors", DOOR, [c.sensor_id for c in CONTACTS]),
              ("Item sensors", ITEM, [i.sensor_id for i in ITEMS])]
    cols = [0.03, 0.40, 0.72]
    for (title, color, ids), cx in zip(groups, cols):
        fig.text(cx, 0.345, title, color=color, fontsize=14, fontweight="bold", va="top")
        for k, sid in enumerate(ids):
            fig.text(cx, 0.315 - k * 0.0195, f"{sid}   {NAMES[sid]}", color=LINE,
                     fontsize=11.5, va="top", family="monospace")
    p2 = os.path.join(OUT_DIR, "blueprint.png")
    fig.savefig(p2, dpi=130, facecolor=BG)
    fig.savefig(p2.replace(".png", ".pdf"), facecolor=BG)
    plt.close(fig)
    print("wrote", p1, p2, p2.replace(".png", ".pdf"))


if __name__ == "__main__":
    main()

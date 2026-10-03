"""The one chart that tells the pen-picking story: how far does the instruction move the arm?

For every attempt, the offline probe swaps the instruction (colour word, or ring position) on
the same frame and measures how far shoulder_pan moves. Switching pens needs the arm to cross the
midpoint between them: 21.7 deg. Values are the numbers recorded in PROJECT_LOG.md.

    uv run --no-sync python my_contributions/tools/make_story_chart.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "media" / "instruction_strength.png"
BAR_DEG = 21.7

# (label, degrees, how the instruction was given, whole-chunk measure?)
# Early runs were only measured with the whole-chunk average, which understates the late-chunk
# number by ~1.4x; even scaled up, none comes near the bar.
RUNS = [
    ("Full fine-tune (3 pens)", 1.9, "word", True),
    ("Lower learning rate", 0.4, "word", True),
    ("Freeze the vision-language model", 1.8, "word", True),
    ("LoRA adapters", 0.95, "word", True),
    ("Bigger model (π0.5)", 0.05, "word", True),
    ("One-word prompts", 2.70, "word", False),
    ("Hide joint state (dropout)", 4.70, "word", False),
    ("Contrastive loss + dropout", 8.91, "word", False),
    ("Contrastive loss", 12.26, "word", False),
    ("Contrastive loss + trimmed data", 19.42, "word", False),
    ("Green ring on the pen", 30.03, "ring", False),
]

# Validated categorical slots 1-2 (dataviz reference palette, light mode) + its text/surface tokens.
COL = {"word": "#2a78d6", "ring": "#eb6834"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = list(range(len(RUNS)))[::-1]
    for y, (label, deg, kind, whole) in zip(ys, RUNS, strict=True):
        ax.barh(y, deg, height=0.55, color=COL[kind], edgecolor=SURFACE, linewidth=1)
        ax.text(deg + 0.4, y, f"{deg:.1f}°" + ("*" if whole else ""), va="center", fontsize=9, color=INK2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1}, zorder=3)
    ax.set_yticks(list(ys), [r[0] for r in RUNS], fontsize=9.5, color=INK)

    ax.axvline(BAR_DEG, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(BAR_DEG + 0.4, len(RUNS) - 0.35, "needed to switch pens (21.7°)", fontsize=9, color=INK, va="bottom")

    # Robot outcomes, on the two checkpoints that went on the arm, next to their values.
    ax.text(BAR_DEG + 0.8, ys[RUNS.index(next(r for r in RUNS if r[1] == 12.26))],
            "robot: same pen every time", va="center", fontsize=8.5, color=INK2)
    ax.text(30.03 + 4.2, ys[-1], "\u2190 robot: right pen 7/7,\n     picked up 6/7", va="center", fontsize=8.5,
            color=INK, fontweight="bold")

    ax.set_xlim(0, 44)
    ax.set_xlabel("How far changing the instruction swings the arm (shoulder pan, degrees)", fontsize=9.5, color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("word", "ring")]
    ax.legend(handles, ["instruction = a colour word in the prompt", "instruction = a green ring in the image"],
              loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Ten ways to make the arm listen to a word, then one that worked", x=0.02, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.005, "* early runs measured over the whole action chunk, which reads ~1.4× lower; "
             "still far below the line.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

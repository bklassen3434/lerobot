"""Compare the live overhead camera against frames from the training dataset.

Usage (from the workspace root, venv activated):
    python my_contributions/tools/camera_check.py
    python my_contributions/tools/camera_check.py --camera 1 --refs .context/camera_check/wrist

Keys:
    e   toggle green edge outline of the reference on top of the live view (best for aligning)
    m   cycle mode: edges -> blend -> side-by-side -> difference
    [ ] blend weight down / up
    n p next / previous reference frame
    s   save a snapshot of the current window
    q   quit
"""

import argparse
import glob
import sys

import cv2
import numpy as np

MODES = ["edges", "blend", "side", "diff"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index (0 = top on this rig)")
    ap.add_argument("--refs", default=".context/camera_check/top", help="folder of reference PNGs")
    ap.add_argument("--mean", default=".context/camera_check/top_mean.png", help="averaged reference frame")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.refs}/*.png"))
    if args.mean:
        paths = [args.mean] + paths
    refs = [(p, cv2.imread(p)) for p in paths]
    refs = [(p, im) for p, im in refs if im is not None]
    if not refs:
        print(f"no reference images found in {args.refs}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1

    idx, mode, alpha = 0, 0, 0.5
    win = "training (reference) vs live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, live = cap.read()
        if not ok:
            print("camera read failed", file=sys.stderr)
            break
        path, ref = refs[idx]
        ref = cv2.resize(ref, (live.shape[1], live.shape[0]))

        # how different the live view is from the reference, ignoring colour/brightness
        score = float(np.mean(cv2.absdiff(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY),
                                          cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))))

        if MODES[mode] == "edges":
            edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 80, 160)
            view = live.copy()
            view[edges > 0] = (0, 255, 0)
        elif MODES[mode] == "blend":
            view = cv2.addWeighted(ref, alpha, live, 1 - alpha, 0)
        elif MODES[mode] == "side":
            view = np.hstack([ref, live])
        else:
            view = cv2.absdiff(ref, live)

        label = f"{MODES[mode]}  ref={path.split('/')[-1]}  diff={score:5.1f}  [e/m/[/]/n/p/s/q]"
        cv2.putText(view, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            break
        elif k == ord("m"):
            mode = (mode + 1) % len(MODES)
        elif k == ord("e"):
            mode = 0
        elif k == ord("["):
            alpha = max(0.0, alpha - 0.05)
        elif k == ord("]"):
            alpha = min(1.0, alpha + 0.05)
        elif k == ord("n"):
            idx = (idx + 1) % len(refs)
        elif k == ord("p"):
            idx = (idx - 1) % len(refs)
        elif k == ord("s"):
            out = ".context/camera_check/snapshot.png"
            cv2.imwrite(out, view)
            print(f"saved {out}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

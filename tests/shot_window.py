"""Dev helper: capture the overlay window by its real rect (DPI-aware)."""
import ctypes
import ctypes.wintypes as wt
import sys

import mss

ctypes.windll.shcore.SetProcessDpiAwareness(2)
u = ctypes.windll.user32

hwnd = u.FindWindowW(None, "Conference Copilot")
if not hwnd:
    sys.exit("overlay window not found")

aff = ctypes.c_ulong(0)
u.GetWindowDisplayAffinity(hwnd, ctypes.byref(aff))

r = wt.RECT()
u.GetWindowRect(hwnd, ctypes.byref(r))
print(f"hwnd={hwnd} rect=({r.left},{r.top},{r.right},{r.bottom}) "
      f"size={r.right - r.left}x{r.bottom - r.top} affinity={aff.value:#x}")

x, y = r.left, r.top
w, h = r.right - r.left, r.bottom - r.top
pad = 6
mon = {"left": max(0, x - pad), "top": max(0, y - pad),
       "width": w + 2 * pad, "height": h + 2 * pad}
with mss.mss() as sct:
    img = sct.grab(mon)
    out = sys.argv[1] if len(sys.argv) > 1 else "window_shot.png"
    mss.tools.to_png(img.rgb, img.size, output=out)
print("saved", out)

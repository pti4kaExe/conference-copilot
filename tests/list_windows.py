"""Dev helper: list visible top-level windows with owning PID and rect."""
import ctypes
import ctypes.wintypes as wt

u = ctypes.windll.user32
rows = []


@ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
def cb(hwnd, _lparam):
    if not u.IsWindowVisible(hwnd):
        return True
    length = u.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    u.GetWindowTextW(hwnd, buf, length + 1)
    pid = wt.DWORD()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    rect = wt.RECT()
    u.GetWindowRect(hwnd, ctypes.byref(rect))
    rows.append((pid.value, buf.value, (rect.left, rect.top, rect.right, rect.bottom)))
    return True


u.EnumWindows(cb, 0)
for pid, title, rect in rows:
    print(f"{pid:>7}  {rect[0]:>5},{rect[1]:>5} {rect[2] - rect[0]:>4}x{rect[3] - rect[1]:<4} {title!r}")

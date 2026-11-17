#!/usr/bin/env python3
"""Conference Copilot — live captions + Q&A hints overlay for Windows.

Pipeline:
    system audio (WASAPI loopback)
        -> energy VAD segmenter
        -> faster-whisper (local, CPU int8)
        -> always-on-top overlay (tkinter)
        -> OpenRouter LLM suggestions for questions

Hotkeys (global, work while another app has focus):
    Ctrl+Alt+Q  ask the model for an answer right now
    Ctrl+Alt+H  hide / show the overlay
    Ctrl+Alt+E  toggle "hidden from screen capture" (presenter-notes mode)
    Ctrl+Alt+T  toggle click-through (mouse passes to the window below)
    Ctrl+Alt+S  save the transcript so far
    Ctrl+Alt+X  quit (saves transcript)

Usage:
    python copilot.py                 # normal run
    python copilot.py --selftest 10   # 10 s headless smoke test, prints captions
    python copilot.py --list-devices  # show audio devices
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE, "config.json")

DEFAULTS = {
    "openrouter_api_key": "",
    "answer_model": "openai/gpt-4o-mini",
    "whisper_model": "base",
    "language": "ru",
    "cpu_threads": 4,
    "sample_rate": 16000,
    "opacity": 0.93,
    "width": 580,
    "height": 470,
    "x": None,
    "y": None,
    "hide_from_capture": True,
    "auto_answers": True,
    "answer_cooldown_sec": 7,
    "context_chars": 3000,
}

QUESTION_STARTS = (
    "как ", "что ", "почему", "зачем", "когда", "где ", "кто ", "сколько",
    "может", "можете", "расскаж", "объясн", "поясн", "какой", "какая", "какие",
    "what", "how", "why", "when", "where", "who", "which", "can you", "could you",
)

SYS_PROMPT = (
    "Ты — суфлёр на конференции. Тебе дают последние реплики транскрипта. "
    "Если прозвучал вопрос — выдай готовый ответ, который можно сразу произнести "
    "вслух: 2–3 коротких пункта, не больше 60 слов, без приветствий, без «спасибо "
    "за вопрос», без markdown и заголовков. Если это утверждение — 1–2 тезиса, "
    "которые усилят или уточнят его. Пиши на языке транскрипта (обычно русский). "
    "Только текст ответа, без пояснений о том, что ты делаешь."
)


# --------------------------------------------------------------------------- config
def ensure_config() -> None:
    """First run: create the local config.json from the committed example."""
    if os.path.exists(CFG_PATH):
        return
    example = os.path.join(BASE, "config.example.json")
    if os.path.exists(example):
        shutil.copyfile(example, CFG_PATH)
        print("[config] создал config.json из config.example.json — "
              "впиши openrouter_api_key, иначе подсказки работать не будут")


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH, "r", encoding="utf-8") as fh:
                cfg.update(json.load(fh))
        except Exception as exc:  # noqa: BLE001
            print(f"[config] не смог прочитать config.json: {exc}")
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        cfg["openrouter_api_key"] = env_key
    return cfg


def save_config_position(x: int, y: int, w: int, h: int) -> None:
    try:
        raw = {}
        if os.path.exists(CFG_PATH):
            with open(CFG_PATH, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        raw.update({"x": x, "y": y, "width": w, "height": h})
        with open(CFG_PATH, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- audio
def find_loopback():
    import soundcard as sc

    speaker = sc.default_speaker()
    try:
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        if mic is not None:
            return mic, speaker
    except Exception:  # noqa: BLE001
        pass
    for mic in sc.all_microphones(include_loopback=True):
        if getattr(mic, "isloopback", False):
            return mic, speaker
    raise RuntimeError(
        "Не нашёл loopback-устройство. Проверь, что есть активный вывод звука "
        "(наушники/колонки), и запусти `python copilot.py --list-devices`."
    )


def list_devices() -> None:
    import soundcard as sc

    print("Speakers:")
    for spk in sc.all_speakers():
        print(f"  - {spk.name}  (default={spk.name == sc.default_speaker().name})")
    print("Microphones (включая loopback):")
    for mic in sc.all_microphones(include_loopback=True):
        print(f"  - {mic.name}  loopback={getattr(mic, 'isloopback', False)}")


class AudioCapture(threading.Thread):
    """Reads WASAPI loopback in 100 ms blocks and pushes float32 mono blocks."""

    def __init__(self, audio_q: queue.Queue, events: queue.Queue, cfg: dict):
        super().__init__(daemon=True, name="audio")
        self.audio_q = audio_q
        self.events = events
        self.sr = int(cfg["sample_rate"])
        self.block_sec = 0.1
        self.running = threading.Event()
        self.running.set()

    def run(self) -> None:
        import warnings

        try:
            import soundcard as sc  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", f"soundcard не установлен: {exc}"))
            return
        # occasional buffer starves on Windows are harmless — keep the console clean
        warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")
        try:
            mic, speaker = find_loopback()
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", str(exc)))
            return

        self.events.put(("status", f"Слушаю: {speaker.name}"))
        frames = max(1, int(self.sr * self.block_sec))
        try:
            with mic.recorder(samplerate=self.sr, channels=2, blocksize=frames) as rec:
                while self.running.is_set():
                    data = rec.record(numframes=frames)
                    if data is None or len(data) == 0:
                        continue
                    mono = data.mean(axis=1) if getattr(data, "ndim", 1) > 1 else data
                    try:
                        self.audio_q.put_nowait(mono.astype(np.float32, copy=False))
                    except queue.Full:
                        pass
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", f"Ошибка захвата звука: {exc}"))


class Segmenter(threading.Thread):
    """Energy VAD: cuts speech into 0.6–9 s chunks for the recognizer."""

    MIN_SPEECH_SEC = 0.4
    MAX_SPEECH_SEC = 5.0
    SILENCE_TO_CUT = 0.45

    def __init__(self, audio_q: queue.Queue, seg_q: queue.Queue, events: queue.Queue, cfg: dict):
        super().__init__(daemon=True, name="vad")
        self.audio_q = audio_q
        self.seg_q = seg_q
        self.events = events
        self.sr = int(cfg["sample_rate"])
        self.running = threading.Event()
        self.running.set()
        self.noise = 0.003
        self.level = 0.0

    def run(self) -> None:
        block = 0.1
        silence_need = int(self.SILENCE_TO_CUT / block)
        max_blocks = int(self.MAX_SPEECH_SEC / block)
        min_blocks = int(self.MIN_SPEECH_SEC / block)

        buf: list[np.ndarray] = []
        silence = 0
        speaking = False

        while self.running.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-9)
            self.level = rms
            if not speaking:
                # adapt noise floor while nobody talks
                self.noise = 0.995 * self.noise + 0.005 * min(rms, 0.05)
            threshold = max(0.0045, self.noise * 4.0)

            if rms > threshold:
                speaking = True
                silence = 0
                buf.append(chunk)
            elif speaking:
                silence += 1
                buf.append(chunk)
                if silence >= silence_need:
                    self._emit(buf, min_blocks)
                    buf, silence, speaking = [], 0, False
                continue

            if speaking and len(buf) >= max_blocks:
                self._emit(buf, min_blocks)
                buf, silence, speaking = [], 0, False

    def _emit(self, buf: list[np.ndarray], min_blocks: int) -> None:
        if len(buf) < min_blocks:
            return
        data = np.concatenate(buf).astype(np.float32, copy=False)
        # skip near-silent junk
        if float(np.sqrt(np.mean(np.square(data)))) < 0.0035:
            return
        try:
            self.seg_q.put_nowait(data)
        except queue.Full:
            pass


# --------------------------------------------------------------------------- stt
class Transcriber(threading.Thread):
    def __init__(self, seg_q: queue.Queue, events: queue.Queue, cfg: dict):
        super().__init__(daemon=True, name="stt")
        self.seg_q = seg_q
        self.events = events
        self.cfg = cfg
        self.running = threading.Event()
        self.running.set()
        self.model = None

    def run(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", f"faster-whisper не установлен: {exc}"))
            return

        name = str(self.cfg.get("whisper_model") or "base")
        threads = int(self.cfg.get("cpu_threads") or 8)
        self.events.put(("status", f"Загружаю модель {name} (первый раз — скачивание)..."))
        try:
            self.model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", f"Не смог загрузить модель {name}: {exc}"))
            return
        self.events.put(("status", "Готов. Слушаю звук системы"))

        lang = (self.cfg.get("language") or "").strip() or None
        tail = ""
        while self.running.is_set():
            try:
                audio = self.seg_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                segments, _info = self.model.transcribe(
                    audio,
                    language=lang,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    initial_prompt=(tail[-300:] or None),
                    without_timestamps=True,
                )
                text = "".join(seg.text for seg in segments).strip()
            except Exception as exc:  # noqa: BLE001
                self.events.put(("error", f"STT: {exc}"))
                continue
            if not text:
                continue
            tail = (tail + " " + text)[-600:]
            self.events.put(("transcript", text))
            self.events.put(("latency", f"{time.time() - t0:.1f}s"))


# --------------------------------------------------------------------------- llm
class AnswerEngine(threading.Thread):
    """Watches the transcript and asks OpenRouter for a ready-to-say answer."""

    def __init__(self, events: queue.Queue, cfg: dict):
        super().__init__(daemon=True, name="llm")
        self.events = events
        self.cfg = cfg
        self.running = threading.Event()
        self.running.set()
        self.req_q: queue.Queue[str] = queue.Queue()
        self.context: list[str] = []
        self.last_answer_at = 0.0
        self.lock = threading.Lock()

    # called from UI thread on every new caption line
    def on_line(self, text: str) -> None:
        with self.lock:
            self.context.append(text)
            self.context = self.context[-40:]
            context = " ".join(self.context)
        if not self.cfg.get("auto_answers"):
            return
        low = text.lower().strip()
        is_question = "?" in text or low.startswith(QUESTION_STARTS)
        if not is_question:
            return
        if time.time() - self.last_answer_at < float(self.cfg["answer_cooldown_sec"]):
            return
        self.force((" ".join(self.context)[-int(self.cfg["context_chars"]):]))

    def force(self, context: str | None = None) -> None:
        key = (self.cfg.get("openrouter_api_key") or "").strip()
        if not key:
            self.events.put(("answer", "⚠️ Нет API-ключа OpenRouter. "
                                       "Впиши его в config.json → openrouter_api_key."))
            return
        with self.lock:
            ctx = context or " ".join(self.context)[-int(self.cfg["context_chars"]):]
        if not ctx.strip():
            self.events.put(("answer", "Пока нечего отвечать — тишина в эфире."))
            return
        self.last_answer_at = time.time()
        self.events.put(("answer_status", "думаю…"))
        try:
            self.req_q.put_nowait(ctx)
        except queue.Full:
            pass

    def run(self) -> None:
        import requests

        url = "https://openrouter.ai/api/v1/chat/completions"
        key = (self.cfg.get("openrouter_api_key") or "").strip()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Conference Copilot",
        }
        while self.running.is_set():
            try:
                ctx = self.req_q.get(timeout=0.3)
            except queue.Empty:
                continue
            payload = {
                "model": self.cfg.get("answer_model") or "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": f"Последние реплики:\n{ctx}\n\nДай ответ."},
                ],
                "max_tokens": 220,
                "temperature": 0.35,
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=40)
                if resp.status_code != 200:
                    self.events.put(("answer", f"OpenRouter {resp.status_code}: {resp.text[:300]}"))
                    continue
                data = resp.json()
                answer = data["choices"][0]["message"]["content"].strip()
                self.events.put(("answer", answer))
            except Exception as exc:  # noqa: BLE001
                self.events.put(("answer", f"Ошибка запроса: {exc}"))


# --------------------------------------------------------------------------- win32
user32 = ctypes.windll.user32

WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x0001, 0x0002, 0x4000
VK = {"Q": 0x51, "H": 0x48, "E": 0x45, "T": 0x54, "S": 0x53, "X": 0x58}
WDA_NONE, WDA_EXCLUDEFROMCAPTURE = 0x00000000, 0x00000011
GWL_EXSTYLE = -20
WS_EX_LAYERED, WS_EX_TRANSPARENT = 0x00080000, 0x00000020


def dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        try:
            user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


class HotkeyThread(threading.Thread):
    def __init__(self, bindings: dict[str, tuple[int, str, object]]):
        super().__init__(daemon=True, name="hotkeys")
        self.bindings = bindings
        self._cb: dict[int, object] = {}

    def run(self) -> None:
        msg = wt.MSG()
        idx = 1
        for key, (mods, vk, cb) in self.bindings.items():
            if user32.RegisterHotKey(None, idx, mods | MOD_NOREPEAT, VK[vk]):
                self._cb[idx] = cb
            idx += 1
        if not self._cb:
            return
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY:
                cb = self._cb.get(int(msg.wParam))
                if cb is not None:
                    try:
                        cb()
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()


# --------------------------------------------------------------------------- ui
class Overlay:
    def __init__(self, cfg: dict, events: queue.Queue, engine: AnswerEngine):
        import tkinter as tk
        from tkinter import scrolledtext

        self.tk = tk
        self.cfg = cfg
        self.events = events
        self.engine = engine
        self.lines: list[str] = []
        self.hidden = False
        self.passthrough = False
        self.exclude = bool(cfg.get("hide_from_capture", True))
        self.running = True

        root = tk.Tk()
        self.root = root
        root.title("Conference Copilot")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", float(cfg.get("opacity", 0.93)))
        root.configure(bg="#0e1318")

        w = int(cfg.get("width") or 580)
        h = int(cfg.get("height") or 470)
        # geometry is in logical px while the screen may be at 125/150 % DPI
        scale = root.winfo_fpixels("1i") / 96.0
        w, h = int(w * scale), int(h * scale)
        x = cfg.get("x")
        y = cfg.get("y")
        if x is None or y is None:
            x = max(20, root.winfo_screenwidth() - w - int(40 * scale))
            y = int(40 * scale)
        root.geometry(f"{w}x{h}+{int(x)}+{int(y)}")

        outer = tk.Frame(root, bg="#2b3a49", padx=1, pady=1)
        outer.pack(fill="both", expand=True)
        body = tk.Frame(outer, bg="#0e1318")
        body.pack(fill="both", expand=True)

        # header ------------------------------------------------------------
        head = tk.Frame(body, bg="#0e1318", height=26)
        head.pack(fill="x", padx=10, pady=(6, 2))
        self.dot = tk.Label(head, text="●", fg="#3ddc84", bg="#0e1318", font=("Segoe UI", 9))
        self.dot.pack(side="left")
        tk.Label(head, text=" CONF COPILOT", fg="#8fb3cc", bg="#0e1318",
                 font=("Segoe UI Semibold", 9)).pack(side="left")
        self.status = tk.Label(head, text="запуск…", fg="#6d7f8f", bg="#0e1318",
                               font=("Segoe UI", 8), anchor="e")
        self.status.pack(side="right")
        for wid in (head, self.dot, self.status):
            wid.bind("<ButtonPress-1>", self._drag_start)
            wid.bind("<B1-Motion>", self._drag_move)

        # transcript --------------------------------------------------------
        tk.Label(body, text="ТРАНСКРИПТ", fg="#4d6172", bg="#0e1318",
                 font=("Segoe UI", 7, "bold"), anchor="w").pack(fill="x", padx=10)
        self.txt = scrolledtext.ScrolledText(
            body, wrap="word", bg="#0e1318", fg="#dbe6f0", insertbackground="#dbe6f0",
            relief="flat", font=("Segoe UI", 10), height=12, padx=6, pady=4,
            borderwidth=0, highlightthickness=0,
        )
        self.txt.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        self.txt.configure(state="disabled")

        # answer ------------------------------------------------------------
        tk.Label(body, text="ПОДСКАЗКА", fg="#4d6172", bg="#0e1318",
                 font=("Segoe UI", 7, "bold"), anchor="w").pack(fill="x", padx=10)
        self.ans = tk.Text(
            body, wrap="word", bg="#101c18", fg="#9fe8b0", relief="flat",
            font=("Segoe UI", 10), height=7, padx=8, pady=6,
            borderwidth=0, highlightthickness=0,
        )
        self.ans.pack(fill="both", expand=False, padx=6, pady=(0, 4))
        self.ans.configure(state="disabled")

        # footer ------------------------------------------------------------
        tk.Label(
            body,
            text="^⌥Q ответ   ^⌥H скрыть   ^⌥E захват   ^⌥T клик-сквозь   ^⌥S сохранить   ^⌥X выход",
            fg="#44586a", bg="#0e1318", font=("Segoe UI", 7), anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 6))

        self.text_widget = self.txt
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.after(120, self._poll)
        root.after(400, self._apply_affinity)

        HotkeyThread({
            "answer": (MOD_CONTROL | MOD_ALT, "Q", lambda: self.root.after(0, self._force_answer)),
            "hide": (MOD_CONTROL | MOD_ALT, "H", lambda: self.root.after(0, self.toggle_hide)),
            "excl": (MOD_CONTROL | MOD_ALT, "E", lambda: self.root.after(0, self.toggle_exclude)),
            "thru": (MOD_CONTROL | MOD_ALT, "T", lambda: self.root.after(0, self.toggle_passthrough)),
            "save": (MOD_CONTROL | MOD_ALT, "S", lambda: self.root.after(0, self.save_transcript)),
            "quit": (MOD_CONTROL | MOD_ALT, "X", lambda: self.root.after(0, self.quit)),
        }).start()

    # -- window helpers ----------------------------------------------------
    def _hwnd(self) -> int:
        hwnd = user32.GetParent(self.root.winfo_id())
        return hwnd or self.root.winfo_id()

    def _apply_affinity(self) -> None:
        try:
            user32.SetWindowDisplayAffinity(
                self._hwnd(), WDA_EXCLUDEFROMCAPTURE if self.exclude else WDA_NONE)
        except Exception:  # noqa: BLE001
            pass

    def toggle_exclude(self) -> None:
        self.exclude = not self.exclude
        self._apply_affinity()
        self._set_status("скрыт из захвата" if self.exclude else "виден в захвате")

    def toggle_passthrough(self) -> None:
        self.passthrough = not self.passthrough
        hwnd = self._hwnd()
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if self.passthrough:
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
        else:
            style &= ~WS_EX_TRANSPARENT
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        self._set_status("клик-сквозь" if self.passthrough else "клики захватываются")

    def toggle_hide(self) -> None:
        self.hidden = not self.hidden
        self.root.withdraw() if self.hidden else self.root.deiconify()
        if not self.hidden:
            self.root.attributes("-topmost", True)

    def _drag_start(self, ev) -> None:
        self._dragging = (ev.x_root, ev.y_root,
                          self.root.winfo_x(), self.root.winfo_y())

    def _drag_move(self, ev) -> None:
        if getattr(self, "_dragging", None) is None:
            return
        sx, sy, wx, wy = self._dragging
        self.root.geometry(f"+{wx + ev.x_root - sx}+{wy + ev.y_root - sy}")

    # -- events ------------------------------------------------------------
    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        if self.running:
            self.root.after(120, self._poll)

    def _handle(self, kind: str, payload: str) -> None:
        if kind == "transcript":
            self.lines.append(payload)
            self._append(self.txt, payload + "\n")
            self.engine.on_line(payload)
        elif kind == "answer":
            self._set_answer(payload)
        elif kind == "answer_status":
            self._set_status(payload)
        elif kind == "status":
            self._set_status(payload)
        elif kind == "error":
            self._set_status("ошибка")
            self._set_answer("⚠️ " + payload)
        elif kind == "latency":
            pass

    def _append(self, widget, text: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _set_answer(self, text: str) -> None:
        self.ans.configure(state="normal")
        self.ans.delete("1.0", "end")
        self.ans.insert("end", text)
        self.ans.configure(state="disabled")
        self._set_status("готово" if not text.startswith("⚠️") else "ошибка")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text[:60])

    def _force_answer(self) -> None:
        self.engine.force()

    def save_transcript(self) -> str:
        os.makedirs(os.path.join(BASE, "notes"), exist_ok=True)
        path = os.path.join(BASE, "notes", time.strftime("transcript_%Y%m%d_%H%M%S.txt"))
        header = f"Conference Copilot — {time.strftime('%Y-%m-%d %H:%M')}\n\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + "\n".join(self.lines))
        self._set_status("сохранено")
        return path

    def quit(self) -> None:
        self.save_transcript()
        save_config_position(self.root.winfo_x(), self.root.winfo_y(),
                             self.root.winfo_width(), self.root.winfo_height())
        self.running = False
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Conference Copilot")
    parser.add_argument("--selftest", type=float, default=0.0,
                        help="run headless for N seconds, print captions, exit")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--visible", action="store_true",
                        help="force overlay to be visible in screen capture (for testing)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    dpi_aware()
    ensure_config()
    cfg = load_config()
    if args.visible:
        cfg["hide_from_capture"] = False
    events: queue.Queue = queue.Queue()
    audio_q: queue.Queue = queue.Queue(maxsize=400)
    seg_q: queue.Queue = queue.Queue(maxsize=40)

    capture = AudioCapture(audio_q, events, cfg)
    vad = Segmenter(audio_q, seg_q, events, cfg)
    stt = Transcriber(seg_q, events, cfg)
    engine = AnswerEngine(events, cfg)
    for th in (capture, vad, stt, engine):
        th.start()

    if args.selftest or args.no_ui:
        deadline = time.time() + (args.selftest or 10)
        print(f"[selftest] слушаю {deadline - time.time():.0f} c …")
        while time.time() < deadline:
            try:
                kind, payload = events.get(timeout=0.3)
            except queue.Empty:
                continue
            print(f"[{kind}] {payload}")
        engine.running.clear()
        stt.running.clear()
        return 0

    overlay = Overlay(cfg, events, engine)
    try:
        overlay.root.mainloop()
    except KeyboardInterrupt:
        overlay.quit()
    finally:
        for th in (capture, vad, stt, engine):
            th.running.clear()
    return 0


if __name__ == "__main__":
    sys.exit(main())

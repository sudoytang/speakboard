#!/usr/bin/env python3
"""
Whisperer MVP
按住右 Option 键开始录音，松开后自动转写并通过悬浮窗确认，回车后输入到光标处。
首次运行会自动下载 Whisper 模型。
"""

import ctypes
import os
import queue
import signal
import subprocess
import threading
import time
import numpy as np
import pyperclip
import sounddevice as sd
import mlx_whisper

from AppKit import (
    NSApplication, NSPanel, NSScreen, # pyright: ignore[reportAttributeAccessIssue]
    NSTextField, NSColor, NSFont, # pyright: ignore[reportAttributeAccessIssue]
    NSWindowCollectionBehaviorCanJoinAllSpaces, # pyright: ignore[reportAttributeAccessIssue]
    NSStatusWindowLevel, # pyright: ignore[reportAttributeAccessIssue]
    NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, # pyright: ignore[reportAttributeAccessIssue]
)
from Foundation import NSObject, NSTimer, NSMakeRect, NSWorkspace  # type: ignore[import]
from pynput import keyboard

# ── 配置 ────────────────────────────────────────────────────────────────────
MODEL_REPO = "mlx-community/whisper-medium-mlx"  # 可换 whisper-small-mlx / whisper-large-v3-mlx
SAMPLE_RATE = 16000
HOTKEY = keyboard.Key.alt_r  # 右 Option 键
SILENCE_RMS_THRESHOLD = 0.002  # 低于此值视为静音，跳过转写
# NSWindowStyleMask: Titled=1, Closable=2, NonactivatingPanel=128
_PANEL_STYLE = 1 | 2 | 128
# ─────────────────────────────────────────────────────────────────────────────

recording = False
audio_frames: list[np.ndarray] = []
_main_queue: queue.Queue = queue.Queue()

# 悬浮窗状态
_overlay_active = False
_overlay_prev_app = None
_overlay_text_field = None
_active_panel = None


def _run_on_main(func) -> None:
    """从任意线程将 func 调度到主线程执行。"""
    _main_queue.put(func)


def _sigint_handler(*_) -> None:
    print("\n[whisperer] 收到 SIGINT，发送 SIGTERM 退出…")
    os.kill(os.getpid(), signal.SIGTERM)


class QueueDrainer(NSObject):
    """由 NSTimer 每 50ms 驱动，排空主线程任务队列。"""

    def drain_(self, _timer) -> None:
        # NSApp 会把 SIGINT 设为 SIG_IGN，每次都重装回来
        signal.signal(signal.SIGINT, _sigint_handler)
        while True:
            try:
                _main_queue.get_nowait()()
            except queue.Empty:
                break


def _close_overlay() -> None:
    """关闭悬浮窗（在主线程调用）。"""
    global _active_panel, _overlay_active, _overlay_text_field, _overlay_prev_app
    if _active_panel:
        _active_panel.orderOut_(None)
        _active_panel = None
    _overlay_active = False
    _overlay_text_field = None
    _overlay_prev_app = None


_CG = ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
_CG.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
_CG.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
_CG.CGEventSetFlags.restype = None
_CG.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
_CG.CGEventPost.restype = None
_CG.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
_CG.CFRelease.restype = None
_CG.CFRelease.argtypes = [ctypes.c_void_p]

_APP_SERVICES = ctypes.CDLL('/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices')
_APP_SERVICES.AXIsProcessTrusted.restype = ctypes.c_bool

kCGHIDEventTap = 0


def _check_ax_trusted() -> bool:
    return bool(_APP_SERVICES.AXIsProcessTrusted())


def _post_cmd_v(_pid: int) -> None:
    """向当前最前台 App 发送 Cmd+V（HID 级别，需要辅助功能权限）。"""
    kVK_ANSI_V = 0x09
    kCGEventFlagMaskCommand = 0x100000
    for keydown in (True, False):
        ev = _CG.CGEventCreateKeyboardEvent(None, kVK_ANSI_V, keydown)
        _CG.CGEventSetFlags(ev, kCGEventFlagMaskCommand)
        _CG.CGEventPost(kCGHIDEventTap, ev)
        _CG.CFRelease(ev)


def _confirm_on_main() -> None:
    """读取文本、关闭悬浮窗、粘贴文字（在主线程调用）。"""
    text = str(_overlay_text_field.stringValue()).strip() if _overlay_text_field else ""
    prev = _overlay_prev_app
    bundle_id = str(prev.bundleIdentifier()) if prev else ""  # type: ignore[union-attr]
    pid = int(prev.processIdentifier()) if prev else 0  # type: ignore[union-attr]
    _close_overlay()
    def schedule_paste() -> None:
        if bundle_id:
            subprocess.run(
                ['osascript', '-e', f'tell application id "{bundle_id}" to activate'],
                check=False,
            )
        time.sleep(0.3)
        _run_on_main(lambda: _do_paste(text, pid))
    threading.Thread(target=schedule_paste, daemon=True).start()


def _do_paste(text: str, pid: int) -> None:
    """把文字写入剪贴板，然后向目标 PID 发 Cmd+V。"""
    if not text:
        return
    pyperclip.copy(text)
    time.sleep(0.05)
    print(f"[whisperer] 向 PID={pid} 发送 Cmd+V，文字={text!r}")
    _post_cmd_v(pid)


def show_overlay(text: str, prev_app: object) -> None:
    """在主线程创建并显示悬浮窗。"""
    global _active_panel, _overlay_active, _overlay_prev_app, _overlay_text_field

    if _active_panel:
        _active_panel.orderOut_(None)

    w, h = 520, 90
    fr = NSScreen.mainScreen().frame()
    x = (fr.size.width - w) / 2
    y = fr.size.height / 3

    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, w, h), _PANEL_STYLE, NSBackingStoreBuffered, False
    )
    panel.setTitle_("Whisperer")
    panel.setLevel_(NSStatusWindowLevel)
    panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces)
    panel.setFloatingPanel_(True)

    # 可编辑文本框
    tf = NSTextField.alloc().initWithFrame_(NSMakeRect(10, 45, 500, 35))
    tf.setStringValue_(text)
    tf.setFont_(NSFont.systemFontOfSize_(14))
    tf.setEditable_(True)
    tf.setSelectable_(True)
    panel.contentView().addSubview_(tf)

    # 提示文字
    hint = NSTextField.alloc().initWithFrame_(NSMakeRect(10, 15, 500, 22))
    hint.setStringValue_("↩ 确认输入   ⎋ 取消")
    hint.setFont_(NSFont.systemFontOfSize_(11))
    hint.setEditable_(False)
    hint.setBezeled_(False)
    hint.setDrawsBackground_(False)
    hint.setTextColor_(NSColor.secondaryLabelColor())
    panel.contentView().addSubview_(hint)

    _active_panel = panel
    _overlay_prev_app = prev_app
    _overlay_text_field = tf
    _overlay_active = True

    panel.makeKeyAndOrderFront_(None)
    panel.makeFirstResponder_(tf)


def load_model() -> None:
    if not _check_ax_trusted():
        print("[whisperer] ⚠️  未获得辅助功能权限！粘贴将无法生效。")
        print("[whisperer]    请前往：系统设置 → 隐私与安全性 → 辅助功能，把终端（或 python）加进去，然后重启程序。")
    print(f"[whisperer] 正在预热模型 {MODEL_REPO}（首次运行需要下载）…")
    mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=MODEL_REPO)
    print("[whisperer] 模型加载完毕，按住右 Option 键开始录音。")


def audio_callback(indata: np.ndarray, frames: int, t, status) -> None:
    if recording:
        audio_frames.append(indata.copy())


def transcribe_and_paste(prev_app: object) -> None:
    if not audio_frames:
        return

    t0 = time.perf_counter()
    audio = np.concatenate(audio_frames).flatten()
    duration = len(audio) / SAMPLE_RATE
    print(f"[whisperer] 录音时长 {duration:.2f}s，开始转写…")

    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"[whisperer] 音量 RMS={rms:.4f}")
    if rms < SILENCE_RMS_THRESHOLD:
        print("[whisperer] 未检测到语音（静音）。")
        return

    t1 = time.perf_counter()
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_REPO,
        initial_prompt="以下是普通话的句子。",
    )
    text = str(result["text"]).strip()
    detected = result.get("language", "?")
    t2 = time.perf_counter()
    print(f"[whisperer] 转写耗时 {t2 - t1:.2f}s，总耗时 {t2 - t0:.2f}s")

    if not text:
        print("[whisperer] 未检测到语音。")
        return

    print(f"[whisperer] [{detected}] {text}")
    _run_on_main(lambda: show_overlay(text, prev_app))


def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    global recording, audio_frames, _overlay_active
    if _overlay_active:
        if key == keyboard.Key.enter:
            _overlay_active = False
            _run_on_main(_confirm_on_main)
        elif key == keyboard.Key.esc:
            _overlay_active = False
            _run_on_main(_close_overlay)
        return
    if key == HOTKEY and not recording:
        recording = True
        audio_frames = []
        print("[whisperer] 录音中…")


def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
    global recording
    if key == HOTKEY and recording:
        recording = False
        # 松开按键时立即记录当前焦点 App
        prev_app = NSWorkspace.sharedWorkspace().frontmostApplication()
        threading.Thread(target=transcribe_and_paste, args=(prev_app,), daemon=True).start()


def main() -> None:
    load_model()

    # Accessory 策略：无 Dock 图标，显示窗口时不抢夺其他 App 的焦点
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app.finishLaunching()

    # 每 50ms 排空主线程任务队列
    drainer = QueueDrainer.alloc().init()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.05, drainer, b'drain:', None, True
    )


    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    print("[whisperer] 按 Ctrl+C 退出。")
    with stream:
        listener.start()
        app.run()  # 主线程在此阻塞，直到 terminate_() 被调用
        listener.stop()

    print("\n[whisperer] 已退出。")


if __name__ == "__main__":
    main()

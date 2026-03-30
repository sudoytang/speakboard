import argparse
import sys

# Per-backend watchdog defaults: (min_timeout_seconds, timeout_multiplier)
_WATCHDOG_DEFAULTS = {
    "mlx": (5.0, 1.0),
    "cpu": (10.0, 5.0),
}


def _make_transcriber(backend: str | None, min_timeout: float | None, multiplier: float | None):
    from .transcribe import WatchdogTranscriber
    resolved = backend or ("mlx" if sys.platform == "darwin" else "cpu")

    if resolved == "mlx":
        from .transcribe import MLXWhisperTranscriber
        factory = MLXWhisperTranscriber
    elif resolved == "cpu":
        from .transcribe import CPUWhisperTranscriber
        factory = CPUWhisperTranscriber
    else:
        raise ValueError(f"Unknown backend '{resolved}'. Choose 'mlx' or 'cpu'.")

    default_min, default_mult = _WATCHDOG_DEFAULTS[resolved]
    return WatchdogTranscriber(
        factory,
        min_timeout_seconds=min_timeout if min_timeout is not None else default_min,
        timeout_multiplier=multiplier if multiplier is not None else default_mult,
    )


def _add_backend_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", choices=["mlx", "cpu"], default=None,
        help="Transcription backend: 'mlx' (Apple Silicon) or 'cpu' (PyTorch). "
             "Defaults to 'mlx' on macOS, 'cpu' elsewhere.",
    )


def _add_watchdog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--watchdog-min-seconds", type=float, default=None, metavar="SECONDS",
        help="Watchdog minimum timeout in seconds. "
             "Default: 5 for mlx, 10 for cpu.",
    )
    parser.add_argument(
        "--watchdog-multiplier", type=float, default=None, metavar="X",
        help="Watchdog timeout = max(min, audio_duration * X). "
             "Default: 1 for mlx, 5 for cpu.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="speakboard")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Standalone CLI mode: hotkey-driven recording and clipboard copy (macOS only)")
    _add_backend_arg(run_parser)
    _add_watchdog_args(run_parser)

    serve_parser = sub.add_parser("serve", help="HTTP server mode: POST /transcribe to get transcription JSON")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    _add_backend_arg(serve_parser)
    _add_watchdog_args(serve_parser)

    args = parser.parse_args()

    if args.command == "run":
        if sys.platform != "darwin":
            raise NotImplementedError(f"CLI mode ('run') is only supported on macOS, not '{sys.platform}'.")
        from .cli import Whisperer
        transcriber = _make_transcriber(args.backend, args.watchdog_min_seconds, args.watchdog_multiplier)
        Whisperer(transcriber).run()

    elif args.command == "serve":
        from .server import start_server
        transcriber = _make_transcriber(args.backend, args.watchdog_min_seconds, args.watchdog_multiplier)
        start_server(transcriber, host=args.host, port=args.port)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

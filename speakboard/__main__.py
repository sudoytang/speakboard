import argparse
import sys


def _make_transcriber(backend: str | None):
    resolved = backend or ("mlx" if sys.platform == "darwin" else "cpu")
    if resolved == "mlx":
        from .transcribe import MLXWhisperTranscriber
        return MLXWhisperTranscriber()
    elif resolved == "cpu":
        from .transcribe import CPUWhisperTranscriber
        return CPUWhisperTranscriber()
    else:
        raise ValueError(f"Unknown backend '{resolved}'. Choose 'mlx' or 'cpu'.")


def _add_backend_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", choices=["mlx", "cpu"], default=None,
        help="Transcription backend: 'mlx' (Apple Silicon) or 'cpu' (PyTorch). "
             "Defaults to 'mlx' on macOS, 'cpu' elsewhere.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="speakboard")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Standalone CLI mode: hotkey-driven recording and clipboard copy (macOS only)")
    _add_backend_arg(run_parser)

    serve_parser = sub.add_parser("serve", help="HTTP server mode: POST /transcribe to get transcription JSON")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    _add_backend_arg(serve_parser)

    args = parser.parse_args()

    if args.command == "run":
        if sys.platform != "darwin":
            raise NotImplementedError(f"CLI mode ('run') is only supported on macOS, not '{sys.platform}'.")
        from .cli import Whisperer
        Whisperer(_make_transcriber(args.backend)).run()

    elif args.command == "serve":
        from .server import start_server
        start_server(_make_transcriber(args.backend), host=args.host, port=args.port)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

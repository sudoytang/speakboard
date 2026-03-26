import sys

from .transcribe import MLXWhisperTranscriber
from .core import Whisperer


def main() -> None:
    if sys.platform != "darwin":
        raise NotImplementedError(f"Platform '{sys.platform}' is not yet supported.")

    Whisperer(MLXWhisperTranscriber()).run()


if __name__ == "__main__":
    main()

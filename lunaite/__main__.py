import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lunaite.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    main()

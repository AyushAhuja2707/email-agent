import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


def main():
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

"""Write or check receipts/INDEX.md (see tinytms/receipts_index.py).

    python scripts/receipts_index.py            # (re)write receipts/INDEX.md
    python scripts/receipts_index.py --check    # CI: exit 1 if INDEX.md is stale or a receipt fails to verify
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytms.receipts_index import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

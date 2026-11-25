"""Run a portable polynomial moment verification manifest."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .receipts import replay_manifest


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        if args.output.resolve() == args.manifest.resolve():
            parser.error("manifest and output paths must be different")
        args.output.unlink(missing_ok=True)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        receipts = replay_manifest(manifest)
    except (ValueError, OSError) as error:
        receipts = [{"verdict": "FAIL", "reason": str(error)}]
    rendered = json.dumps(receipts, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                delete=False) as temporary:
            temporary.write(rendered)
        try:
            Path(temporary.name).replace(args.output)
        finally:
            Path(temporary.name).unlink(missing_ok=True)
    return int(any(not receipt.get("passed", receipt["verdict"] in ("CERTIFIED", "REFUTED"))
                   for receipt in receipts))


if __name__ == "__main__":
    raise SystemExit(main())

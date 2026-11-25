"""Run a portable polynomial moment verification manifest."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from .receipts import replay_manifest


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _same_input_and_output(manifest, output):
    if output.resolve() == manifest.resolve():
        return True
    try:
        return output.samefile(manifest)
    except FileNotFoundError:
        return False


def _write_receipts(output, rendered):
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(rendered)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        try:
            if _same_input_and_output(args.manifest, args.output):
                parser.error("manifest and output paths must be different")
            args.output.unlink(missing_ok=True)
        except (OSError, RuntimeError) as error:
            print(f"Could not prepare receipt output: {error}", file=sys.stderr)
            return 1
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        receipts = replay_manifest(manifest)
    except (ValueError, OSError, RecursionError) as error:
        receipts = [{"verdict": "FAIL", "reason": str(error)}]
    except Exception as error:
        receipts = [{"verdict": "FAIL", "checker_error": True,
                     "reason": f"{type(error).__name__}: {error}"}]
    rendered = json.dumps(receipts, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        try:
            _write_receipts(args.output, rendered)
        except OSError as error:
            print(f"Could not write receipts: {error}", file=sys.stderr)
            return 1
    return int(any(not receipt.get("passed", receipt["verdict"] in ("CERTIFIED", "REFUTED"))
                   for receipt in receipts))


if __name__ == "__main__":
    raise SystemExit(main())

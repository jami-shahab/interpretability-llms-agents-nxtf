"""MEP writer and reader utilities."""

import json
from pathlib import Path
from typing import Iterator

from .schema import MEP


def write_mep(mep: MEP, out_dir: str) -> str:
    """Serialise a MEP to JSON and write to disk.

    Parameters
    ----------
    mep : MEP
        The completed evaluation packet.
    out_dir : str
        Directory to write ``<case_id>.json`` into.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    case_id = mep.case.case_id if mep.case else mep.run_id
    path = Path(out_dir) / f"{case_id}.json"
    with open(path, "w") as f:
        json.dump(mep.to_dict(), f, indent=2, default=str)
    return str(path.resolve())


def iter_meps(mep_dir: str) -> Iterator[dict]:
    """Yield each MEP JSON file from a directory as a plain dict.

    Parameters
    ----------
    mep_dir : str
        Directory containing ``*.json`` MEP files.

    Yields
    ------
    dict
        Parsed MEP dict.
    """
    for p in sorted(Path(mep_dir).glob("*.json")):
        try:
            with open(p) as f:
                yield json.load(f)
        except Exception as exc:
            print(f"  Warning: could not read {p.name}: {exc}")

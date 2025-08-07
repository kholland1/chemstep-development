from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Iterable


@dataclass
class RoundRecord:
    round_n:          int
    n_docked:         int
    n_hits:           int
    n_beacons:        int
    last_beacon_dist: float
    mintd_thresh:     float


class Bookkeeper:
    """
    Tracks ChemSTEP progress and streams two .df files (space-separated):
      • run_summary.df – per-round metrics
      • beacons.df     – one row per beacon (round, id, dist)
    """

    def __init__(self, out_dir: str, smi_id_prefix: str = "CSLB"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.smi_id_prefix = smi_id_prefix

        # create headers
        (self.dir / "run_summary.df").write_text(
            "round n_docked n_hits n_beacons last_beacon_dist mintd_thresh\n"
        )
        (self.dir / "beacons.df").write_text(
            "round beacon_id beacon_score beacon_min_dist\n"
        )

    # ------------------------------------------------------------------ #
    # logging helpers                                                    #
    # ------------------------------------------------------------------ #
    def log_round(
        self,
        round_n: int,
        n_docked: int,
        n_hits: int,
        mintd_thresh: float,
        beacon_ids: Iterable[int],
        beacon_scores: Iterable[float],
        beacon_dists: Iterable[float],
    ):
        beacon_ids = list(beacon_ids)
        beacon_dists = list(beacon_dists)
        n_beacons = len(beacon_ids)
        last_dist = beacon_dists[-1] if beacon_dists else -1.0

        # append to run_summary.df
        with open(self.dir / "run_summary.df", "a") as f:
            f.write(
                f"{round_n} {n_docked} {n_hits} {n_beacons} "
                f"{last_dist:.3f} {mintd_thresh:.4f}\n"
            )

        # append all beacons of this round
        if n_beacons:
            with open(self.dir / "beacons.df", "a") as fb:
                for bid, bscore, dist in zip(beacon_ids, beacon_scores, beacon_dists):
                    fb.write(f"{round_n} {bid} {bscore} {dist:.3f}\n")

    def write_round_docked(
        self, round_n: int, smiles: List[str], abs_ids: List[int]
    ):
        """Write <SMILES> <prefixed-ID> for manual (or fallback) docking."""
        fn = self.dir / f"smi_round_{round_n}.smi"
        with open(fn, "w") as f:
            for smi, zid in zip(smiles, abs_ids):
                f.write(f"{smi} {self.smi_id_prefix}{zid}\n")

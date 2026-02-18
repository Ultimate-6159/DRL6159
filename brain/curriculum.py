"""
Apex Predator — Curriculum Learning Scheduler
================================================
Progressively trains on market data from easy → hard:

Phase 1: TRENDING only        (clear directional moves)
Phase 2: TRENDING + RANGING   (add mean-reverting segments)
Phase 3: ALL regimes           (full difficulty including high-vol/uncertain)

Uses the existing RegimeClassifier's regime labels to filter data
into contiguous segments appropriate for each difficulty level.
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np

from config.settings import CurriculumConfig, MarketRegime

logger = logging.getLogger("apex_predator.curriculum")


class CurriculumScheduler:
    """
    Schedules training data by progressive difficulty using regime labels.

    Each phase filters the full dataset to only include contiguous segments
    where the market regime matches the allowed set. The PPO model is
    continued (not reset) across phases, so knowledge transfers.

    ANTI-FORGETTING: Each phase includes a small % of OTHER regimes
    to prevent catastrophic forgetting and maintain generalization.
    """

    # Regime indices (matching MarketRegime enum order)
    TRENDING = 0
    MEAN_REVERTING = 1
    HIGH_VOLATILITY = 2
    UNCERTAIN = 3

    # Phase definitions: name → allowed regime indices
    # vaccine_pct = percent of "other" regimes to mix in for anti-forgetting
    PHASES = [
        {"name": "Phase 1: Trending Only",
         "regimes": [0],           # TRENDING
         "vaccine_regimes": [1],   # Mix in some MEAN_REVERTING
         "vaccine_pct": 0.15,      # 15% vaccine
         "description": "Easy — clear directional moves + 15% vaccine"},
        {"name": "Phase 2: Trend + Range",
         "regimes": [0, 1],        # TRENDING + MEAN_REVERTING
         "vaccine_regimes": [2, 3], # Mix in some HIGH_VOL/UNCERTAIN
         "vaccine_pct": 0.10,      # 10% vaccine
         "description": "Medium — add mean-reverting + 10% vaccine"},
        {"name": "Phase 3: All Regimes",
         "regimes": [0, 1, 2, 3],  # ALL
         "vaccine_regimes": [],
         "vaccine_pct": 0.0,
         "description": "Hard — full market complexity"},
    ]

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.phase_ratios = config.phase_ratios
        self.min_segment_length = config.min_segment_length

    def get_phase_schedule(self, total_timesteps: int) -> List[Dict]:
        """
        Compute the timestep budget for each curriculum phase.

        Args:
            total_timesteps: Total training timesteps across all phases.

        Returns:
            List of dicts with 'name', 'regimes', 'timesteps' per phase.
        """
        schedule = []
        for i, phase in enumerate(self.PHASES):
            ratio = self.phase_ratios[i] if i < len(self.phase_ratios) else 0.0
            ts = int(total_timesteps * ratio)
            schedule.append({
                "name": phase["name"],
                "description": phase["description"],
                "regimes": phase["regimes"],
                "vaccine_regimes": phase.get("vaccine_regimes", []),
                "vaccine_pct": phase.get("vaccine_pct", 0.0),
                "timesteps": ts,
                "phase_idx": i,
            })
        logger.info("Curriculum schedule: %s",
                     [(s["name"], f"{s['timesteps']:,} steps") for s in schedule])
        return schedule

    def filter_data_by_regimes(
        self,
        bars: np.ndarray,
        features: np.ndarray,
        regimes: np.ndarray,
        spreads: np.ndarray,
        atrs: np.ndarray,
        allowed_regimes: List[int],
        vaccine_regimes: List[int] = None,
        vaccine_pct: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter training data to only include contiguous segments
        where the regime is in the allowed set.

        ANTI-FORGETTING: Mixes in a small percentage of "vaccine" regimes
        to prevent catastrophic forgetting of how to handle them.

        Args:
            bars: (N, 4) OHLC array
            features: (N, feat_dim) feature matrix
            regimes: (N,) regime index array
            spreads: (N,) spread array
            atrs: (N,) ATR array
            allowed_regimes: List of regime indices to include
            vaccine_regimes: List of regime indices to mix in as "vaccine"
            vaccine_pct: Percentage of vaccine data to mix in (0.0-1.0)

        Returns:
            Filtered (bars, features, regimes, spreads, atrs)
        """
        N = len(bars)
        vaccine_regimes = vaccine_regimes or []

        if N == 0:
            return bars, features, regimes, spreads, atrs

        # If all regimes are allowed, return full data
        if set(allowed_regimes) >= {0, 1, 2, 3}:
            logger.info("All regimes allowed — using full dataset (%d bars)", N)
            return bars, features, regimes, spreads, atrs

        # ═══ MAIN REGIME DATA ═══
        mask = np.isin(regimes, allowed_regimes)
        segments = self._find_contiguous_segments(mask)
        valid_segments = [
            (start, end) for start, end in segments
            if (end - start) >= self.min_segment_length
        ]

        if not valid_segments:
            logger.warning(
                "No segments ≥ %d bars for regimes %s. "
                "Falling back to full dataset.",
                self.min_segment_length, allowed_regimes,
            )
            return bars, features, regimes, spreads, atrs

        # Collect main data
        bar_chunks = []
        feat_chunks = []
        regime_chunks = []
        spread_chunks = []
        atr_chunks = []
        main_bars = 0

        for start, end in valid_segments:
            bar_chunks.append(bars[start:end])
            feat_chunks.append(features[start:end])
            regime_chunks.append(regimes[start:end])
            spread_chunks.append(spreads[start:end])
            atr_chunks.append(atrs[start:end])
            main_bars += end - start

        # ═══ VACCINE DATA (Anti-Forgetting) ═══
        vaccine_bars = 0
        if vaccine_regimes and vaccine_pct > 0:
            vaccine_mask = np.isin(regimes, vaccine_regimes)
            vaccine_segments = self._find_contiguous_segments(vaccine_mask)
            valid_vaccine = [
                (start, end) for start, end in vaccine_segments
                if (end - start) >= self.min_segment_length // 2  # Shorter segments OK for vaccine
            ]

            if valid_vaccine:
                # Calculate how many vaccine bars we need
                target_vaccine_bars = int(main_bars * vaccine_pct / (1 - vaccine_pct))

                # Sample vaccine segments
                for start, end in valid_vaccine:
                    if vaccine_bars >= target_vaccine_bars:
                        break
                    bar_chunks.append(bars[start:end])
                    feat_chunks.append(features[start:end])
                    regime_chunks.append(regimes[start:end])
                    spread_chunks.append(spreads[start:end])
                    atr_chunks.append(atrs[start:end])
                    vaccine_bars += end - start

                logger.info(
                    "Vaccine data: added %d bars of regimes %s (%.1f%% of phase)",
                    vaccine_bars, vaccine_regimes, 
                    100 * vaccine_bars / (main_bars + vaccine_bars)
                )

        # Concatenate all chunks
        filtered_bars = np.concatenate(bar_chunks, axis=0)
        filtered_features = np.concatenate(feat_chunks, axis=0)
        filtered_regimes = np.concatenate(regime_chunks, axis=0)
        filtered_spreads = np.concatenate(spread_chunks, axis=0)
        filtered_atrs = np.concatenate(atr_chunks, axis=0)

        total_bars = main_bars + vaccine_bars
        logger.info(
            "Curriculum filter: regimes=%s + vaccine=%s → %d bars (%.1f%% of original)",
            allowed_regimes, vaccine_regimes if vaccine_bars > 0 else "none",
            total_bars, 100 * total_bars / N,
        )

        return (filtered_bars, filtered_features, filtered_regimes,
                filtered_spreads, filtered_atrs)

    @staticmethod
    def _find_contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find contiguous runs of True in a boolean mask.

        Returns:
            List of (start, end) tuples (end is exclusive).
        """
        segments = []
        in_segment = False
        start = 0

        for i in range(len(mask)):
            if mask[i] and not in_segment:
                start = i
                in_segment = True
            elif not mask[i] and in_segment:
                segments.append((start, i))
                in_segment = False

        if in_segment:
            segments.append((start, len(mask)))

        return segments

    def log_phase_stats(
        self,
        phase_name: str,
        bars: np.ndarray,
        regimes: np.ndarray,
    ):
        """Log statistics about the data in a curriculum phase."""
        regime_counts = np.bincount(regimes.astype(int), minlength=4)
        regime_names = ["TRENDING", "MEAN_REV", "HIGH_VOL", "UNCERTAIN"]
        stats = ", ".join(
            f"{name}={count}" for name, count in zip(regime_names, regime_counts)
        )
        logger.info("[%s] %d bars — %s", phase_name, len(bars), stats)

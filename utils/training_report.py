# -*- coding: utf-8 -*-
"""
Apex Predator — Training Report Logger
=======================================
บันทึกทุกอย่างที่เกี่ยวกับการเทรนลงไฟล์ เพื่อวิเคราะห์และปรับปรุง
"""

import os
import json
import logging
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger("apex_predator.training_report")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class TrainingReport:
    """
    บันทึกข้อมูลการเทรนทั้งหมด:
    - Config ที่ใช้
    - ข้อมูลที่เทรน (จำนวน bars, regime distribution)
    - ผลระหว่างเทรน (progress)
    - Action distribution (ตรวจจับ bias)
    - ผล Evaluation สุดท้าย
    """

    def __init__(self, output_dir: str = "reports/", experiment_name: str = None):
        self.report_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{experiment_name}_" if experiment_name else "training_report_"
        self.report_path = os.path.join(output_dir, f"{prefix}{self.timestamp}.json")
        self.text_report_path = os.path.join(output_dir, f"{prefix}{self.timestamp}.txt")

        self.report: Dict[str, Any] = {
            "meta": {
                "experiment_name": experiment_name,
                "timestamp": self.timestamp,
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "status": "running",
            },
            "config": {},
            "data_info": {},
            "imitation_learning": {},
            "curriculum_phases": [],
            "training_progress": [],
            "action_distribution": {
                "buy_count": 0,
                "sell_count": 0,
                "hold_count": 0,
                "buy_pct": 0.0,
                "sell_pct": 0.0,
                "hold_pct": 0.0,
                "bias_warning": None,
            },
            "evaluation": {
                "episodes": [],
                "average": {},
            },
            "warnings": [],
            "recommendations": [],
        }

    # ═══════════════════════════════════════════
    # CONFIG LOGGING
    # ═══════════════════════════════════════════

    def log_config(self, config, extra_params: Dict[str, Any] = None) -> None:
        """บันทึก configuration ทั้งหมด"""
        try:
            self.report["config"] = {
                "mt5": {
                    "symbol": config.mt5.symbol,
                    "timeframe": config.mt5.timeframe,
                },
                "drl": {
                    "algorithm": config.drl.algorithm,
                    "learning_rate": config.drl.learning_rate,
                    "gamma": config.drl.gamma,
                    "clip_range": config.drl.clip_range,
                    "n_steps": config.drl.n_steps,
                    "batch_size": config.drl.batch_size,
                    "n_epochs": config.drl.n_epochs,
                    "ent_coef": config.drl.ent_coef,
                    "total_timesteps": config.drl.total_timesteps,
                },
                "reward": {
                    "profit_reward": config.reward.profit_reward,
                    "loss_penalty": config.reward.loss_penalty,
                    "hold_penalty": config.reward.hold_penalty,
                    "max_hold_steps": config.reward.max_hold_steps,
                    "drawdown_penalty": config.reward.drawdown_penalty,
                },
                "risk": {
                    "max_risk_per_trade": config.risk.max_risk_per_trade,
                    "atr_multiplier": config.risk.atr_multiplier,
                    "tp_ratio": config.risk.tp_ratio,
                    "trade_cooldown_sec": config.risk.trade_cooldown_sec,
                },
                "curriculum": {
                    "enabled": config.curriculum.enabled,
                    "phase_ratios": config.curriculum.phase_ratios,
                },
                "imitation": {
                    "enabled": config.imitation.enabled,
                    "epochs": config.imitation.epochs,
                    "lookahead_bars": config.imitation.lookahead_bars,
                },
            }
            # Add extra parameters if provided
            if extra_params:
                self.report["config"]["extra"] = extra_params
            logger.info("Config logged to report")
        except Exception as e:
            logger.error("Failed to log config: %s", e)

    # ═══════════════════════════════════════════
    # DATA INFO LOGGING
    # ═══════════════════════════════════════════

    def log_data_info(
        self,
        total_bars: int,
        valid_bars: int,
        start_date: str = None,
        end_date: str = None,
        regime_counts: Dict[str, int] = None,
    ) -> None:
        """บันทึกข้อมูลเกี่ยวกับ dataset"""
        regime_counts = regime_counts or {}
        self.report["data_info"] = {
            "total_bars": total_bars,
            "valid_bars": valid_bars,
            "date_start": start_date,
            "date_end": end_date,
            "regime_distribution": regime_counts,
        }

        # Check for regime imbalance
        total_regimes = sum(regime_counts.values())
        if total_regimes > 0:
            for regime, count in regime_counts.items():
                pct = count / total_regimes
                if pct > 0.7:
                    warning = f"⚠️ Data is {pct:.0%} {regime} - may cause bias"
                    self.report["warnings"].append(warning)
                    logger.warning(warning)

        logger.info("Data info logged: %d bars, %d valid", total_bars, valid_bars)

    # ═══════════════════════════════════════════
    # IMITATION LEARNING LOGGING
    # ═══════════════════════════════════════════

    def log_imitation(self, stats: Dict[str, Any]) -> None:
        """บันทึกผล Imitation Learning (รับ dict จาก ImitationPreTrainer)"""
        if stats.get("skipped"):
            self.report["imitation_learning"] = {
                "skipped": True,
                "reason": stats.get("reason", "unknown"),
            }
            logger.info("Imitation learning skipped: %s", stats.get("reason"))
            return

        total_samples = stats.get("total_samples", 0)
        buy_samples = stats.get("buy_samples", 0)
        sell_samples = stats.get("sell_samples", 0)
        hold_samples = stats.get("hold_samples", 0)
        final_loss = stats.get("final_loss", 0.0)
        accuracy = stats.get("accuracy", 0.0)

        self.report["imitation_learning"] = {
            "total_samples": total_samples,
            "buy_samples": buy_samples,
            "sell_samples": sell_samples,
            "hold_samples": hold_samples,
            "buy_pct": buy_samples / max(total_samples, 1),
            "sell_pct": sell_samples / max(total_samples, 1),
            "hold_pct": hold_samples / max(total_samples, 1),
            "final_loss": final_loss,
            "accuracy": accuracy,
        }

        # Check for imbalanced labels
        total = buy_samples + sell_samples + hold_samples
        if total > 0:
            buy_pct = buy_samples / total
            sell_pct = sell_samples / total
            if buy_pct > 0.6:
                self.report["warnings"].append(f"⚠️ Imitation labels heavily biased to BUY ({buy_pct:.0%})")
            if sell_pct > 0.6:
                self.report["warnings"].append(f"⚠️ Imitation labels heavily biased to SELL ({sell_pct:.0%})")

        logger.info("Imitation learning logged: %d samples, accuracy=%.2f%%", total_samples, accuracy * 100)

    # ═══════════════════════════════════════════
    # CURRICULUM PHASE LOGGING
    # ═══════════════════════════════════════════

    def log_curriculum_phase(
        self,
        phase_idx: int,
        phase_name: str,
        timesteps: int,
        bars: int,
        regime_counts: Dict[str, int] = None,
        elapsed_sec: float = 0.0,
    ) -> None:
        """บันทึกแต่ละ phase ของ curriculum"""
        self.report["curriculum_phases"].append({
            "phase_idx": phase_idx,
            "phase_name": phase_name,
            "timesteps": timesteps,
            "bars": bars,
            "regime_counts": regime_counts or {},
            "elapsed_sec": elapsed_sec,
        })
        logger.info("Curriculum phase '%s' logged: %d timesteps, %d bars in %.1fs", 
                    phase_name, timesteps, bars, elapsed_sec)

    # ═══════════════════════════════════════════
    # PPO TRAINING METRICS LOGGING
    # ═══════════════════════════════════════════

    def log_ppo_metrics(
        self,
        timestep: int,
        metrics: Dict[str, float],
    ) -> None:
        """บันทึก PPO metrics (loss, kl, entropy, etc.)"""
        if "ppo_metrics" not in self.report:
            self.report["ppo_metrics"] = []

        entry = {
            "timestep": timestep,
            "loss": metrics.get("train/loss", 0),
            "policy_gradient_loss": metrics.get("train/policy_gradient_loss", 0),
            "value_loss": metrics.get("train/value_loss", 0),
            "entropy_loss": metrics.get("train/entropy_loss", 0),
            "approx_kl": metrics.get("train/approx_kl", 0),
            "clip_fraction": metrics.get("train/clip_fraction", 0),
            "explained_variance": metrics.get("train/explained_variance", 0),
            "learning_rate": metrics.get("train/learning_rate", 0),
        }
        self.report["ppo_metrics"].append(entry)

    # ═══════════════════════════════════════════
    # TRAINING PROGRESS LOGGING
    # ═══════════════════════════════════════════

    def log_training_progress(
        self,
        timestep: int,
        episode: int,
        reward: float,
        win_rate: float,
        total_trades: int,
        action_counts: Dict[int, int],
    ) -> None:
        """บันทึก progress ระหว่างเทรน (เรียกจาก callback)"""
        self.report["training_progress"].append({
            "timestep": timestep,
            "episode": episode,
            "reward": reward,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "action_counts": action_counts,
        })

        # Update action distribution
        buy = action_counts.get(0, 0)
        sell = action_counts.get(1, 0)
        hold = action_counts.get(2, 0)
        total = buy + sell + hold

        self.report["action_distribution"]["buy_count"] += buy
        self.report["action_distribution"]["sell_count"] += sell
        self.report["action_distribution"]["hold_count"] += hold

        if total > 0:
            self.report["action_distribution"]["buy_pct"] = buy / total
            self.report["action_distribution"]["sell_pct"] = sell / total
            self.report["action_distribution"]["hold_pct"] = hold / total

    # ═══════════════════════════════════════════
    # EVALUATION LOGGING
    # ═══════════════════════════════════════════

    def log_evaluation_episode(self, episode: int, stats: Dict[str, Any]) -> None:
        """บันทึกผล evaluation แต่ละ episode (รับ dict จาก env.get_stats())"""
        self.report["evaluation"]["episodes"].append({
            "episode": episode,
            "total_trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate", 0),
            "pnl": stats.get("total_pnl", 0),
            "sharpe": stats.get("sharpe_ratio", 0),
            "final_balance": stats.get("final_balance", 0),
        })

    def log_evaluation_summary(self, all_stats: List[Dict[str, Any]]) -> None:
        """บันทึกสรุปผล evaluation จาก list ของ stats"""
        import numpy as np
        if not all_stats:
            return

        avg_trades = np.mean([s.get("total_trades", 0) for s in all_stats])
        avg_win_rate = np.mean([s.get("win_rate", 0) for s in all_stats])
        avg_pnl = np.mean([s.get("total_pnl", 0) for s in all_stats])
        avg_sharpe = np.mean([s.get("sharpe_ratio", 0) for s in all_stats])

        self.report["evaluation"]["average"] = {
            "avg_trades": avg_trades,
            "avg_win_rate": avg_win_rate,
            "avg_pnl": avg_pnl,
            "avg_sharpe": avg_sharpe,
        }

        # Generate recommendations
        self._generate_recommendations()

    # ═══════════════════════════════════════════
    # ANALYSIS & RECOMMENDATIONS
    # ═══════════════════════════════════════════

    def _generate_recommendations(self) -> None:
        """วิเคราะห์และสร้างคำแนะนำ"""
        recs = []

        # Check action distribution bias
        ad = self.report["action_distribution"]
        total_actions = ad["buy_count"] + ad["sell_count"] + ad["hold_count"]
        if total_actions > 0:
            buy_pct = ad["buy_count"] / total_actions
            sell_pct = ad["sell_count"] / total_actions
            hold_pct = ad["hold_count"] / total_actions

            ad["buy_pct"] = buy_pct
            ad["sell_pct"] = sell_pct
            ad["hold_pct"] = hold_pct

            if buy_pct > 0.6:
                ad["bias_warning"] = f"HEAVY BUY BIAS ({buy_pct:.0%})"
                recs.append("🔴 Model มี bias ไปทาง BUY มากเกินไป - ควร retrain ด้วย balanced data")
            elif sell_pct > 0.6:
                ad["bias_warning"] = f"HEAVY SELL BIAS ({sell_pct:.0%})"
                recs.append("🔴 Model มี bias ไปทาง SELL มากเกินไป - ควร retrain ด้วย balanced data")
            elif hold_pct > 0.8:
                ad["bias_warning"] = f"EXCESSIVE HOLD ({hold_pct:.0%})"
                recs.append("🟡 Model HOLD มากเกินไป - ลอง increase entropy coefficient")

        # Check evaluation results
        eval_avg = self.report["evaluation"].get("average", {})
        if eval_avg:
            if eval_avg.get("avg_win_rate", 0) < 0.5:
                recs.append("🔴 Win Rate < 50% - model ยังไม่พร้อม deploy")
            if eval_avg.get("avg_sharpe", 0) < 1.0:
                recs.append("🟡 Sharpe < 1.0 - risk-adjusted return ยังต่ำ")
            if eval_avg.get("avg_pnl", 0) < 0:
                recs.append("🔴 Average PnL เป็นลบ - ห้าม deploy!")

            if eval_avg.get("avg_win_rate", 0) > 0.7 and eval_avg.get("avg_sharpe", 0) > 2.0:
                recs.append("✅ Model พร้อม deploy! Win Rate และ Sharpe ดี")

        self.report["recommendations"] = recs

    # ═══════════════════════════════════════════
    # SAVE REPORT
    # ═══════════════════════════════════════════

    def finalize(self, status: str = "completed") -> str:
        """บันทึก report ลงไฟล์"""
        self.report["meta"]["completed_at"] = datetime.now().isoformat()
        self.report["meta"]["status"] = status

        # Generate final recommendations
        self._generate_recommendations()

        # Save JSON (use NumpyEncoder to handle numpy types)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        # Save human-readable text report
        self._save_text_report()

        logger.info("📊 Training report saved to: %s", self.report_path)
        return self.report_path

    def _save_text_report(self) -> None:
        """สร้าง text report ที่อ่านง่าย"""
        lines = []
        lines.append("=" * 70)
        lines.append("   APEX PREDATOR — TRAINING REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {self.report['meta']['timestamp']}")
        lines.append(f"Status: {self.report['meta']['status']}")
        lines.append("")

        # Config
        lines.append("─" * 70)
        lines.append("CONFIGURATION")
        lines.append("─" * 70)
        cfg = self.report.get("config", {})
        if cfg:
            lines.append(f"Symbol: {cfg.get('mt5', {}).get('symbol', 'N/A')}")
            lines.append(f"Timeframe: {cfg.get('mt5', {}).get('timeframe', 'N/A')}")
            lines.append(f"Algorithm: {cfg.get('drl', {}).get('algorithm', 'N/A')}")
            lines.append(f"Learning Rate: {cfg.get('drl', {}).get('learning_rate', 'N/A')}")
            lines.append(f"Timesteps: {cfg.get('drl', {}).get('total_timesteps', 'N/A'):,}")
        lines.append("")

        # Data Info
        lines.append("─" * 70)
        lines.append("DATA INFO")
        lines.append("─" * 70)
        data = self.report.get("data_info", {})
        if data:
            lines.append(f"Total Bars: {data.get('total_bars', 0):,}")
            lines.append(f"Valid Samples: {data.get('valid_samples', 0):,}")
            lines.append(f"Date Range: {data.get('date_start')} to {data.get('date_end')}")
            lines.append(f"Regime Distribution: {data.get('regime_distribution', {})}")
        lines.append("")

        # Imitation Learning
        il = self.report.get("imitation_learning", {})
        if il:
            lines.append("─" * 70)
            lines.append("IMITATION LEARNING")
            lines.append("─" * 70)
            lines.append(f"Total Samples: {il.get('total_samples', 0):,}")
            lines.append(f"BUY: {il.get('buy_samples', 0)} ({il.get('buy_pct', 0):.1%})")
            lines.append(f"SELL: {il.get('sell_samples', 0)} ({il.get('sell_pct', 0):.1%})")
            lines.append(f"HOLD: {il.get('hold_samples', 0)} ({il.get('hold_pct', 0):.1%})")
            lines.append(f"Accuracy: {il.get('accuracy', 0):.1%}")
            lines.append("")

        # Action Distribution
        lines.append("─" * 70)
        lines.append("ACTION DISTRIBUTION (during training)")
        lines.append("─" * 70)
        ad = self.report.get("action_distribution", {})
        lines.append(f"BUY:  {ad.get('buy_count', 0):>8,} ({ad.get('buy_pct', 0):.1%})")
        lines.append(f"SELL: {ad.get('sell_count', 0):>8,} ({ad.get('sell_pct', 0):.1%})")
        lines.append(f"HOLD: {ad.get('hold_count', 0):>8,} ({ad.get('hold_pct', 0):.1%})")
        if ad.get("bias_warning"):
            lines.append(f"⚠️  {ad['bias_warning']}")
        lines.append("")

        # PPO Metrics Summary
        ppo_metrics = self.report.get("ppo_metrics", [])
        if ppo_metrics:
            lines.append("─" * 70)
            lines.append("PPO TRAINING METRICS (last recorded)")
            lines.append("─" * 70)
            last = ppo_metrics[-1]
            lines.append(f"  Timestep:          {last.get('timestep', 0):,}")
            lines.append(f"  Loss:              {last.get('loss', 0):.4f}")
            lines.append(f"  Policy Loss:       {last.get('policy_gradient_loss', 0):.4f}")
            lines.append(f"  Value Loss:        {last.get('value_loss', 0):.4f}")
            lines.append(f"  Entropy Loss:      {last.get('entropy_loss', 0):.4f}")
            lines.append(f"  Approx KL:         {last.get('approx_kl', 0):.6f}")
            lines.append(f"  Clip Fraction:     {last.get('clip_fraction', 0):.4f}")
            lines.append(f"  Explained Var:     {last.get('explained_variance', 0):.4f}")
            lines.append(f"  Learning Rate:     {last.get('learning_rate', 0):.2e}")
            lines.append(f"  Total Snapshots:   {len(ppo_metrics)}")
            lines.append("")

        # Evaluation Results
        lines.append("─" * 70)
        lines.append("EVALUATION RESULTS")
        lines.append("─" * 70)
        eval_data = self.report.get("evaluation", {})
        for ep in eval_data.get("episodes", []):
            lines.append(
                f"Episode {ep['episode']}: Trades={ep['total_trades']} "
                f"WR={ep['win_rate']:.1%} PnL=${ep['pnl']:.2f} Sharpe={ep['sharpe']:.2f}"
            )
        avg = eval_data.get("average", {})
        if avg:
            lines.append("")
            lines.append("AVERAGE:")
            lines.append(f"  Trades/ep: {avg.get('avg_trades', 0):.0f}")
            lines.append(f"  Win Rate:  {avg.get('avg_win_rate', 0):.1%}")
            lines.append(f"  PnL:       ${avg.get('avg_pnl', 0):.2f}")
            lines.append(f"  Sharpe:    {avg.get('avg_sharpe', 0):.2f}")
        lines.append("")

        # Warnings
        warnings = self.report.get("warnings", [])
        if warnings:
            lines.append("─" * 70)
            lines.append("⚠️  WARNINGS")
            lines.append("─" * 70)
            for w in warnings:
                lines.append(f"  {w}")
            lines.append("")

        # Recommendations
        recs = self.report.get("recommendations", [])
        if recs:
            lines.append("─" * 70)
            lines.append("📋 RECOMMENDATIONS")
            lines.append("─" * 70)
            for r in recs:
                lines.append(f"  {r}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        with open(self.text_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("📄 Text report saved to: %s", self.text_report_path)


# Global instance for easy access
_current_report: Optional[TrainingReport] = None


def get_training_report() -> TrainingReport:
    """Get or create global training report instance."""
    global _current_report
    if _current_report is None:
        _current_report = TrainingReport()
    return _current_report


def new_training_report(output_dir: str = "reports/", experiment_name: str = None) -> TrainingReport:
    """Create a new training report (starts fresh)."""
    global _current_report
    _current_report = TrainingReport(output_dir=output_dir, experiment_name=experiment_name)
    return _current_report

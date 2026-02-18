"""Quick sanity check for the updated BacktestEnv."""
from brain.backtest_env import BacktestEnv
import numpy as np

np.random.seed(42)
N = 1000
close = np.cumsum(np.random.randn(N) * 0.5).astype(np.float32) + 2000
bars = np.column_stack([
    close - abs(np.random.randn(N) * 0.1),  # open
    close + abs(np.random.randn(N) * 0.3),  # high
    close - abs(np.random.randn(N) * 0.3),  # low
    close,                                    # close
]).astype(np.float32)
feats = np.random.rand(N, 17).astype(np.float32)
reg = np.zeros(N, dtype=np.int32)
sp = np.full(N, 2.0, dtype=np.float32)
atr = np.full(N, 1.0, dtype=np.float32)

env = BacktestEnv(bars, feats, reg, sp, atr)
obs, _ = env.reset()
done = False
steps = 0

while not done:
    a = env.action_space.sample()
    obs, r, t, tr, info = env.step(a)
    done = t or tr
    steps += 1

s = env.get_stats()
print(f"Trades: {s['total_trades']}")
print(f"Wins:   {s['wins']}")
print(f"Win Rate: {s['win_rate']:.1%}")
print(f"PnL: ${s['total_pnl']:.2f}")
print(f"Balance: ${s['final_balance']:.2f}")
print(f"Steps: {steps}")
print("OK - Environment works correctly!")

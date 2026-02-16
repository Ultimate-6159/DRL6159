# 🦁 Apex Predator — DRL Trading System

ระบบเทรด Forex อัตโนมัติระดับ "ผู้ล่าสูงสุด" ใช้ Deep Reinforcement Learning สำหรับ Scalping/HFT

## สถาปัตยกรรม 4 Layer

```
Layer 1: The Eyes    → LSTM + Attention (Pattern Memory)
Layer 2: The Brain   → PPO Agent (Decision Engine)  
Layer 3: The Shield  → Risk Manager + Circuit Breaker (Math Only)
Layer 4: Evolution   → Online Learning + Walk-Forward
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure MT5 (Edit `config/settings.py`)
```python
@dataclass
class MT5Config:
    login: int = 12345678          # Your MT5 account
    password: str = "your_pass"    # Your password
    server: str = "YourBroker"     # Broker server
```

### 3. Run (Dry-Run Mode)
```bash
python main.py
```

### 4. Run (Live Trading ⚠️)
```bash
python main.py --live --symbol EURUSD --timeframe M1
```

## 📁 Project Structure

```
DRL6159/
├── config/settings.py        # All parameters in one place
├── core/
│   ├── mt5_connector.py      # MT5 API wrapper
│   ├── data_feed.py          # Real-time data buffer
│   └── feature_engine.py     # Feature extraction & normalization
├── brain/
│   ├── regime_classifier.py  # Market state detection
│   ├── perception.py         # LSTM + Attention encoder
│   ├── drl_agent.py          # PPO/SAC decision engine
│   └── reward.py             # Sharpe-based reward function
├── shield/
│   ├── risk_manager.py       # Position sizing (NO AI — pure math)
│   └── circuit_breaker.py    # Emergency stop mechanism
├── evolution/
│   ├── online_trainer.py     # Incremental learning
│   └── walk_forward.py       # Performance validation
├── memory/
│   └── vector_store.py       # Pattern similarity search
├── main.py                   # Orchestrator
└── tests/                    # Unit & integration tests
```

## ⚙️ CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--live` | off | Enable real trading (default: dry-run) |
| `--symbol` | EURUSD | Trading pair |
| `--timeframe` | M1 | Chart timeframe |
| `--log-level` | INFO | Log verbosity |

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## ⚠️ Disclaimer

ระบบนี้ออกแบบเป็น **foundation/skeleton** สำหรับการพัฒนาต่อ ก่อนใช้เงินจริงต้อง:
1. Train โมเดลด้วยข้อมูลจริง (historical data)
2. Backtest อย่างละเอียด
3. Paper trade อย่างน้อย 1-3 เดือน
4. ใช้ Demo Account ก่อนเสมอ

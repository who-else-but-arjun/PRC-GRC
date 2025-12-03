pip install numpy pandas gymnasium stable-baselines3 torch tqdm matplotlib
```

### Step 2: Prepare Data
Place your raw tick data CSV files in a folder (e.g., `EBX/`):
```
EBX/
├── day1.csv
├── day2.csv
└── day3.csv
```

Each CSV should have columns: `Time, Price`

### Step 3: Run Pipeline

---

## Commands Explained

### Command 1: `python script.py train EBX`

**What Happens:**

1. **Data Resampling** (5-10 mins)
   - Reads tick data from `EBX/` folder
   - Converts to 2-minute OHLC candles
   - Saves to `EBX_2min/` (skips if already exists)
   - Creates `train_days_EBX.txt` and `test_days_EBX.txt`

2. **Indicator Calculation** (10-30 mins)
   - Precomputes 60+ technical indicators for ALL training days
   - Applies 30-minute warmup window (discards first 30 mins of each day)
   - Stores in memory (~10MB per day)

3. **Model Training** (30-120 mins depending on CPU/GPU)
   - Launches parallel environments (CPU cores - 2)
   - Trains PPO model for 600,000 timesteps
   - Prints training progress with tqdm bar
   - Monitors: entropy loss, explained variance, policy loss
   - GPU auto-detects and uses if available

4. **Model Saving** (1 min)
   - Saves trained model: `Models_EBX/ppo_trading_model_EBX.zip`
   - Saves normalization stats: `Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl`
   - Generates training plots: `training_plots/EBX_training_metrics.png`
   - Generates feature info: `feature_info_EBX.txt`

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "All files already processed" | 2-min candles exist in `EBX_2min/` | Delete `EBX_2min/` folder to reprocess, or ignore (training continues) |
| CUDA out of memory | GPU memory exceeded | Set `use_gpu=False` in code, or reduce `batch_size` from 4096 to 2048 |
| Training stuck at 0% | Wrong data format | Check CSV has `Time` and `Price` columns exactly |
| Memory error (RAM) | Too many training days loaded | Reduce training days or increase RAM |
| "n_envs = 0" error | CPU detection failed | Manually set `NUM_PROCS` in PARAMS (e.g., 4) |
| Model very small (~1MB) | Training interrupted | Rerun command (checkpoints may be corrupted) |

---

### Command 2: `python script.py test EBX`

**What Happens:**

1. **Model Loading** (2 mins)
   - Loads trained model from `Models_EBX/ppo_trading_model_EBX.zip`
   - Loads normalization stats from `Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl`
   - Verifies both files exist

2. **Per-Day Backtesting** (2-5 mins per day, depends on test days count)
   - For each test day:
     - Loads 2-min candle data
     - Calculates indicators (with 30-min warmup)
     - Runs model forward with `deterministic=True`
     - Records every trade entry/exit with price and timestamp
     - Calculates daily P&L in basis points (bps)
     - Generates signals (BUY, SELL, EXIT)

3. **Trade Analysis** (1 min)
   - Calculates per-trade PnL
   - Counts winning/losing trades
   - Computes win rate
   - Tracks highest/lowest prices for trailing stops

4. **Output Generation** (2 mins)
   - Saves all signals to CSV: `signals_EBX/day123.csv`
   - Generates price charts: `test_trade_plots/EBX_day_1_day123.png`
   - Calculates equity curve and drawdown
   - Saves equity plot: `test_results/EBX_equity_drawdown.png`
   - Writes report: `test_results/test_results_EBX.txt`
   - **Prints to console:** Trade log with timestamps, prices, positions

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "KeyError: -1" | current_step exceeded data length | Fixed in latest code - use `last_step_idx = current_step - 1` with bounds check |
| Timestamps show wrong time (+34 mins) | Using 30-min candle timestamps on 2-min data | Fixed - now uses original timestamps from `indicators_df_with_index` |
| "VecNormalize file not found" | Didn't run train command first | Run `python script.py train EBX` first |
| All trades losing | Model overtrained on train set (overfitting) | Train on more diverse data or reduce training episodes |
| 0 trades executed | Model learned to always hold | Increase `TRADE_ENTRY_PENALTY` (currently -5) or check reward scaling |
| Wrong number of test days | `test_days_EBX.txt` corrupted | Delete it and retrain |

---

### Command 3: `python script.py test EBX day123`

**What Happens:**

1. **Specific Day Filtering** (1 sec)
   - Searches for `day123` in test file list
   - Only tests that single day (not all test days)
   - Useful for debugging specific days

2. **Backtesting** (Same as Command 2, but only 1 day)
   - Loads model and normalizer
   - Runs backtest on `day123` only
   - Generates trade plot, signals, and metrics
   - Prints detailed trade log to console

3. **Output** (Same outputs as Command 2, but for single day)

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "Day 123 not found" | Day not in test set or naming mismatch | Use exact day name from `test_days_EBX.txt` (e.g., `day123`, not `123` or `day_123`) |
| Pattern matches multiple days | Using substring match (e.g., `day5` matches `day5`, `day50`, `day500`) | Use full day name like `day00005` with leading zeros |
| CSV file has wrong name format | Filename doesn't match expected pattern | Check `signals_EBX/` folder for actual filenames |

---

### Command 4: `python script.py backtest_ebullient EBX`

**What Happens:**

1. **Signal Loading** (1 sec)
   - Finds all CSV files in `signals_EBX/` folder
   - Numerically sorts by day number
   - Validates files exist

2. **Temporary Directory Setup** (1 sec)
   - Creates temporary folder with config.json
   - Copies signal files to temp location
   - Ebullient expects specific directory structure

3. **Backtest Execution** (5-30 mins depending on days and trades)
   - Initializes BacktesterIIT with config
   - Runs Ebullient's market simulator
   - For each signal:
     - EXIT signal → Closes position
     - BUY signal → Opens long (100 shares)
     - SELL signal → Opens short (100 shares)
   - Calculates slippage, fees, fills

4. **Results & Cleanup** (1 min)
   - Prints position summary
   - Displays P&L, drawdown, win rate
   - Cleans up temporary files

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "No signal files found" | Didn't run `test` command first | Run `python script.py test EBX` to generate signals |
| "alpha_research module not found" | Ebullient library not installed | Install proprietary module or skip backtesting |
| "day(\d+)" regex error | Signal file naming doesn't match pattern | Check files are named like `day1.csv`, `day2.csv` (not `day_1.csv`) |
| Config file error | JSON formatting issue | Manually inspect `config.json` created in root |
| "position_map error" | Ebullient API changed | Check BacktesterIIT version compatibility |
| P&L mismatch vs test command | Different transaction cost calculations | Verify TRANSACTION_COST parameter is same in both |

---

## Feature Overview (Short)

| Feature | What It Does |
|---------|-------------|
| **60+ Indicators** | RSI, CCI, CMO, KAMA, Aroon, Heikin-Ashi, Ribbon (gives model context) |
| **2-Min Candles** | High-frequency intraday data (captures micro-trends) |
| **Trailing Stop** | Locks in profits, exits if price drops 4 bps from peak |
| **Hard Stop Loss** | Forced exit if loss hits -4 bps (risk management) |
| **Asymmetric Rewards** | Losses penalized 4x more than profits (teaches risk aversion) |
| **Trade Entry Penalty** | -5 reward per trade (prevents overtrading) |
| **Parallel Training** | Uses all CPU cores for faster convergence |
| **VecNormalize** | Standardizes observations and rewards for neural network stability |
| **Deterministic Testing** | Same model decisions every test run (no randomness) |
| **Signal Export** | Generates CSV for external backtesting platforms |

---

## Common Issues & Global Solutions

### Issue 1: "PARAMS mismatch between training and testing"
**Problem:** You changed stop loss/trailing stop but didn't retrain
**Solution:** 
- Training uses `STOP_LOSS_TR`, `TRAIL_PCT_TR`
- Testing uses `STOP_LOSS_TE`, `TRAIL_PCT_TE` (can be different!)
- Model learns exits based on TRAINING params
- Testing params determine what exits are ENFORCED during test
- If you change testing params, results will differ (but model hasn't relearned)

### Issue 2: "Too many/too few trades"
**Problem:** Model behavior doesn't match expectations
**Solutions:**
- Too many: Increase `TRADE_ENTRY_PENALTY` (currently -5) to -10 or -15
- Too few: Decrease `TRADE_ENTRY_PENALTY` to 0 or -2
- Too many stops: Decrease `STOP_LOSS` from -0.0004 to -0.0002
- Retrain after changing parameters

### Issue 3: "Lookahead bias in indicators"
**Problem:** Using High/Low from same candle as decision
**Current Status:** FIXED in latest code - using only Close prices
**Check:** `candle_high` and `candle_low` should NOT exist, only `candle_close`

### Issue 4: "Results don't reproduce"
**Problem:** Same day gives different results each time
**Cause:** `deterministic=False` or seed mismatch
**Solution:** Verify `deterministic=True` in test command
**Check:** PARAMS['SEED'] = 2 (should be consistent)

---

## File Checklist After Running
```
After train:
✓ Models_EBX/ppo_trading_model_EBX.zip (2-5MB)
✓ Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl (100KB)
✓ feature_info_EBX.txt (50KB)
✓ train_days_EBX.txt (list of days)
✓ test_days_EBX.txt (list of days)
✓ training_plots/EBX_training_metrics.png (chart)
✓ EBX_2min/ (folder with 2-min candles - auto-created)

After test:
✓ test_results/test_results_EBX.txt (report)
✓ test_results/EBX_equity_drawdown.png (equity chart)
✓ test_trade_plots/ (folder with per-day charts)
✓ signals_EBX/ (folder with signal CSVs)

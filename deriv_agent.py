import asyncio
import json
import ssl

import numpy as np
import pandas as pd
import streamlit as st
import websockets

# ==========================================
# PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="Deriv Multiplier SMC Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Deriv Multiplier SMC Trading Agent")

# Sidebar Controls
st.sidebar.header("🔑 Deriv Account Credentials")

api_token = st.sidebar.text_input(
    "Deriv API Token",
    type="password",
    help="Use a token with Trade permissions.",
)

app_id = st.sidebar.text_input(
    "Deriv App ID",
    value="1089",
)

st.sidebar.header("⚙️ Strategy & Multiplier Settings")

symbol = st.sidebar.selectbox(
    "Asset Symbol",
    ["R_100", "R_75", "R_50", "R_25", "1HZ100V"],
    index=0,
    help="Synthetic Volatility Indices",
)

multiplier_val = st.sidebar.number_input(
    "Multiplier Factor",
    min_value=2,
    max_value=500,
    value=3,
)

stake_amount = st.sidebar.number_input(
    "Stake Per Trade ($)",
    min_value=1.0,
    value=2.0,
    step=0.5,
)

trading_mode = st.sidebar.radio(
    "Execution Style",
    ["Scalper (1m)", "Professional Trend (5m)"],
)

st.sidebar.header("🛡️ Profit Protection & Rules")

daily_target = st.sidebar.number_input(
    "Target Daily Profit ($)",
    min_value=5.0,
    value=10.0,
)

max_daily_loss = st.sidebar.number_input(
    "Max Daily Loss Limit ($)",
    min_value=5.0,
    value=10.0,
)

# ==========================================
# SESSION STATE
# ==========================================
if "daily_pnl" not in st.session_state:
    st.session_state.daily_pnl = 0.0

if "profit_lock_active" not in st.session_state:
    st.session_state.profit_lock_active = False

if "trade_history" not in st.session_state:
    st.session_state.trade_history = []


# ==========================================
# TECHNICAL & SMC ANALYSIS ENGINE
# ==========================================
def analyze_smc_market(df):
    """Calculates POC estimate, detects Fair Value Gaps, and outputs a market signal."""
    required_columns = {"open", "high", "low", "close"}

    if not required_columns.issubset(df.columns):
        return {
            "signal": "HOLD",
            "reason": "Required OHLC candle data is missing",
            "poc": None,
            "price": None,
        }

    if len(df) < 30:
        return {
            "signal": "HOLD",
            "reason": "Insufficient candles: at least 30 required",
            "poc": None,
            "price": None,
        }

    df = df.copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(
        drop=True
    )

    if len(df) < 30:
        return {
            "signal": "HOLD",
            "reason": "Insufficient valid candle data",
            "poc": None,
            "price": None,
        }

    # 1. Price distribution / POC
    price_bins = pd.cut(df["close"], bins=10)
    bin_counts = price_bins.value_counts()

    if bin_counts.empty or bin_counts.max() == 0:
        return {
            "signal": "HOLD",
            "reason": "Unable to calculate price distribution",
            "poc": None,
            "price": float(df["close"].iloc[-1]),
        }

    poc_bin = bin_counts.idxmax()
    poc_price = float(poc_bin.mid)
    current_price = float(df["close"].iloc[-1])

    # 2. Fair Value Gap detection
    bullish_fvg = df["low"].iloc[-1] > df["high"].iloc[-3]
    bearish_fvg = df["high"].iloc[-1] < df["low"].iloc[-3]

    # 3. Candlestick confirmation
    prev_candle_bull = df["close"].iloc[-2] > df["open"].iloc[-2]
    curr_candle_bull = df["close"].iloc[-1] > df["open"].iloc[-1]

    # 4. Signal decision
    signal = "HOLD"
    reason = "No clear SMC setup"

    if (
        current_price > poc_price
        and bullish_fvg
        and curr_candle_bull
        and prev_candle_bull
    ):
        signal = "MULTUP"
        reason = f"Bullish FVG and price above estimated POC (${poc_price:.2f})"

    elif current_price < poc_price and bearish_fvg and not curr_candle_bull:
        signal = "MULTDOWN"
        reason = f"Bearish FVG and price below estimated POC (${poc_price:.2f})"

    return {
        "signal": signal,
        "reason": reason,
        "poc": poc_price,
        "price": current_price,
    }


# ==========================================
# DERIV WEBSOCKET EXECUTION ENGINE
# ==========================================
async def execute_trade_ws(websocket, symbol, contract_type, stake, multiplier):
    """Requests a multiplier proposal and executes buy over WebSocket."""
    proposal_req = {
        "proposal": 1,
        "amount": float(stake),
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "symbol": symbol,
        "multiplier": int(multiplier),
    }

    await websocket.send(json.dumps(proposal_req))
    proposal_response = json.loads(await websocket.recv())

    if "error" in proposal_response:
        st.error(
            "Proposal error: "
            + proposal_response["error"].get("message", "Unknown API error")
        )
        return None

    proposal = proposal_response.get("proposal")
    if not proposal or "id" not in proposal:
        st.error("No valid proposal returned.")
        return None

    proposal_id = proposal["id"]
    buy_req = {"buy": proposal_id, "price": float(stake)}

    await websocket.send(json.dumps(buy_req))
    buy_response = json.loads(await websocket.recv())

    if "error" in buy_response:
        st.error(
            "Purchase error: "
            + buy_response["error"].get("message", "Unknown API error")
        )
        return None

    return buy_response


# ==========================================
# DASHBOARD INTERFACE
# ==========================================
col1, col2, col3 = st.columns(3)

col1.metric("Current Daily PnL", f"${st.session_state.daily_pnl:.2f}")
col2.metric(
    "Profit Lock Status",
    "ACTIVE" if st.session_state.profit_lock_active else "INACTIVE",
)
col3.metric("Trading Mode", trading_mode)

st.warning(
    "Trading involves risk. Test on a demo account first. "
    "PnL shown here is strictly local session tracking."
)

start_agent = st.button("🚀 Start Multiplier Agent")

# ==========================================
# AGENT RUNNER VIA NATIVE WEBSOCKETS
# ==========================================
if start_agent:

    if not api_token.strip():
        st.error("Please enter your Deriv API token in the sidebar.")

    elif not app_id.strip():
        st.error("Please enter your Deriv App ID.")

    elif st.session_state.profit_lock_active:
        st.error("Profit lock is active. Reset session state before starting.")

    else:
        st.info("Establishing WebSocket connection with Deriv API...")

        async def run_agent_loop():
            ssl_context = ssl.create_default_context()
            clean_app_id = app_id.strip()

            ws_url = (
                f"wss://ws.derivws.com/websockets/v3?app_id={clean_app_id}"
            )

            try:
                async with websockets.connect(
                    ws_url,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:

                    # 1. Authorize Token
                    await ws.send(json.dumps({"authorize": api_token.strip()}))
                    auth_res = json.loads(await ws.recv())

                    if "error" in auth_res:
                        st.error(
                            "Authorization failed: "
                            + auth_res["error"].get("message", "Unknown API error")
                        )
                        return

                    st.success("Successfully authenticated with Deriv via WebSocket!")

                    # 2. Fetch Candle Data
                    granularity = 60 if "1m" in trading_mode else 300
                    ticks_req = {
                        "ticks_history": symbol,
                        "adjust_start_time": 1,
                        "count": 50,
                        "end": "latest",
                        "style": "candles",
                        "granularity": granularity,
                    }

                    await ws.send(json.dumps(ticks_req))
                    candles_res = json.loads(await ws.recv())

                    if "error" in candles_res:
                        st.error(
                            "Candle fetch failed: "
                            + candles_res["error"].get("message", "Unknown error")
                        )
                        return

                    candles = candles_res.get("candles", [])

                    if not candles:
                        st.warning("No historical candles returned.")
                        return

                    df = pd.DataFrame(candles)

                    # 3. Analyze Market
                    smc_res = analyze_smc_market(df)

                    st.subheader("Market Analysis Output")

                    if (
                        smc_res["price"] is not None
                        and smc_res["poc"] is not None
                    ):
                        st.write(
                            f"**Current Price:** ${smc_res['price']:.2f} | "
                            f"**Volume POC:** ${smc_res['poc']:.2f}"
                        )
                    else:
                        st.warning(smc_res["reason"])

                    st.write(
                        f"**Signal:** `{smc_res['signal']}` | "
                        f"**Reason:** {smc_res['reason']}"
                    )

                    # 4. Check Safeguards
                    pnl = st.session_state.daily_pnl

                    if pnl >= daily_target:
                        st.session_state.profit_lock_active = True
                        st.warning("Daily profit target reached. Trading halted.")
                        return

                    if pnl <= -max_daily_loss:
                        st.error("Daily loss limit hit. Trading halted.")
                        return

                    # 5. Execute Trade
                    if smc_res["signal"] in ("MULTUP", "MULTDOWN"):
                        st.write(
                            f"Executing signal: **{smc_res['signal']}** "
                            f"at {multiplier_val}x multiplier..."
                        )

                        trade_result = await execute_trade_ws(
                            ws,
                            symbol,
                            smc_res["signal"],
                            stake_amount,
                            multiplier_val,
                        )

                        if trade_result is not None:
                            st.subheader("Trade API Response")
                            st.json(trade_result)
                            st.session_state.trade_history.append(trade_result)

                    else:
                        st.info("No qualifying SMC signal. No trade executed.")

            except Exception as e:
                st.error(f"WebSocket Error: {e}")

        asyncio.run(run_agent_loop())

# ==========================================
# TRADE HISTORY
# ==========================================
st.subheader("Trade Submission History")

if st.session_state.trade_history:
    st.dataframe(
        pd.json_normalize(st.session_state.trade_history),
        use_container_width=True,
    )
else:
    st.info("No trade submissions recorded in this session.")

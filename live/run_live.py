from __future__ import annotations
import json
import sys
from broker.ccxt_binance import CCXTBinanceBroker
from engine import LiveEngine

def main():
    cfg_path = "config_live.json" if len(sys.argv) < 2 else sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    broker = CCXTBinanceBroker(
        exchange_id=cfg.get("exchange", "binanceusdm"),
        api_key=cfg["api_key"],
        api_secret=cfg["api_secret"],
        testnet=bool(cfg.get("testnet", True)),
        mode=cfg.get("mode", "futures")
    )
    engine = LiveEngine(broker, cfg)
    engine.run()

if __name__ == "__main__":
    main()

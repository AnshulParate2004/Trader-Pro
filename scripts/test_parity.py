import json
import numpy as np
import pandas as pd
import joblib

def eval_node(node, feat_map, feat_names):
    if "leaf_value" in node:
        return node["leaf_value"]
    feat_name = feat_names[node["split_feature"]]
    val = feat_map.get(feat_name, np.nan)
    if val is None or np.isnan(val):
        default_left = node.get("default_left", True)
        return eval_node(node["left_child"], feat_map, feat_names) if default_left else eval_node(node["right_child"], feat_map, feat_names)
    if val <= node["threshold"]:
        return eval_node(node["left_child"], feat_map, feat_names)
    else:
        return eval_node(node["right_child"], feat_map, feat_names)

def predict_ts_evaluator(model_json, feat_map):
    feat_names = model_json["features"]
    raw_long = sum(eval_node(t["tree_structure"], feat_map, feat_names) for t in model_json["trees_long"])
    raw_short = sum(eval_node(t["tree_structure"], feat_map, feat_names) for t in model_json["trees_short"])
    p_long = 1.0 / (1.0 + np.exp(-raw_long))
    p_short = 1.0 / (1.0 + np.exp(-raw_short))
    return p_long, p_short

def run_parity_test():
    # 1. Load models
    with open(r"D:\Projects_Main\Trade-Pro\Trader-Pro-Dashboard\src\models\lgbm_1h.json") as f:
        ts_model = json.load(f)
        
    py_model_dict = joblib.load(r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\model\1h\final_model_BTCUSDT_1h.joblib")
    py_model_long = py_model_dict["model_long"]
    py_model_short = py_model_dict["model_short"]
    features = py_model_dict["features"]
    
    # 2. Load last 200 closed 1H candles
    df = pd.read_parquet(r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\data\1h\BTCUSDT_1h.parquet")
    df = df.tail(200).copy()
    
    # Compute features for testing
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    
    # Simple feature mock/real mapping
    X_test = pd.DataFrame(index=df.index)
    for col in features:
        if col.startswith("ret_"):
            n = int(col.split("_")[1])
            X_test[col] = df["close"].pct_change(n).fillna(0)
        elif col == "atr_pct":
            X_test[col] = (df["high"] - df["low"]) / df["close"]
        else:
            X_test[col] = 0.0 # fallback for non-price features in test
            
    # Run predictions
    py_pL = py_model_long.predict_proba(X_test[features])[:, 1]
    py_pS = py_model_short.predict_proba(X_test[features])[:, 1]
    
    ts_pL = []
    ts_pS = []
    for idx, row in X_test[features].iterrows():
        f_map = row.to_dict()
        pL, pS = predict_ts_evaluator(ts_model, f_map)
        ts_pL.append(pL)
        ts_pS.append(pS)
        
    ts_pL = np.array(ts_pL)
    ts_pS = np.array(ts_pS)
    
    diff_L = np.abs(py_pL - ts_pL)
    diff_S = np.abs(py_pS - ts_pS)
    
    max_diff_L = np.max(diff_L)
    max_diff_S = np.max(diff_S)
    
    cutoff = 0.582
    py_signals = (py_pL >= cutoff).astype(int) - (py_pS >= cutoff).astype(int)
    ts_signals = (ts_pL >= cutoff).astype(int) - (ts_pS >= cutoff).astype(int)
    
    mismatches = np.sum(py_signals != ts_signals)
    
    print(f"Tested {len(df)} 1H bars:")
    print(f"Max absolute probability difference (Long):  {max_diff_L:.2e}")
    print(f"Max absolute probability difference (Short): {max_diff_S:.2e}")
    print(f"Signal mismatches across 200 candles: {mismatches}/200")
    
if __name__ == "__main__":
    run_parity_test()

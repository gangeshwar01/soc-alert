import streamlit as st
import pandas as pd
import numpy as np
import io, time, json
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🚨 AI Alert Analysis & Reporting", layout="wide")
st.title("🚨 AI Automation: Real-Time Alert Analysis & Report Generation")
st.caption("Uploads/API → Clean → Classify risk & attack-type → Anomaly detect → Correlate → Export Excel")

# -------------------- Helpers --------------------
def add_log(msg):
    ts = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")

if "logs" not in st.session_state:
    st.session_state.logs = []

def to_datetime_smart(df):
    # Find a time-like column, convert to datetime
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["time", "date", "timestamp", "ts", "flow start"]):
            d = pd.to_datetime(df[c], errors="coerce", utc=True)
            if d.notna().sum() > 0:
                return c, d
    # fallback: no time
    return None, None

def detect_ip_cols(df):
    src = None; dst = None
    for c in df.columns:
        cl = c.lower()
        if src is None and ("src" in cl or "source" in cl or cl in ["ip","src ip","source ip","client ip","src_ip"]):
            if "ip" in cl or "addr" in cl or "ip" in c:
                src = c
        if dst is None and ("dst" in cl or "dest" in cl or "destination" in cl or cl in ["dst ip","destination ip","server ip","dst_ip"]):
            if "ip" in cl or "addr" in cl or "ip" in c:
                dst = c
    # fallback: single generic 'ip'
    if src is None and "ip" in [c.lower() for c in df.columns]:
        src = [c for c in df.columns if c.lower()=="ip"][0]
    return src, dst

def detect_port_col(df):
    for c in df.columns:
        if "port" in c.lower():
            return c
    return None

def clean_numeric(df):
    X = df.select_dtypes(include=["number"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

# ---- Fallback attack-type classifier (rules) ----
def rule_attack_type(row, colmap):
    # Try keyword-based label if exists
    for cname in ["Label","label","Attack","attack","attack_type","Attack Type"]:
        if cname in row.index:
            val = str(row[cname]).lower()
            if any(k in val for k in ["bruteforce","ftp-brute","ssh-brute"]): return "BruteForce"
            if "port" in val and "scan" in val: return "PortScan"
            if "dos" in val or "ddos" in val: return "DoS/DDoS"
            if "web" in val or "xss" in val or "sql" in val: return "WebAttack"
            if "benign" in val: return "Benign"
    # Heuristics using flows
    dst_port = None
    if colmap.get("port") and pd.notna(row[colmap["port"]]):
        try:
            dst_port = int(str(row[colmap["port"]]).strip())
        except:
            dst_port = None
    fwd = row.get(" Total Fwd Packets", row.get("Total Fwd Packets", np.nan))
    bwd = row.get(" Total Backward Packets", row.get("Total Backward Packets", np.nan))
    pps = row.get(" Flow Packets/s", row.get("Flow Packets/s", np.nan))
    byps = row.get("Flow Bytes/s", np.nan)
    # PortScan: many distinct destination ports overall (handled in aggregation), but row-wise hint:
    if dst_port in [21,22,23,25,80,443,3389,53] and pd.notna(pps) and pps>500:
        return "PortScan"
    # DoS/DDoS: high packet rate/bytes per sec
    if (pd.notna(pps) and pps>2000) or (pd.notna(byps) and byps>1e6):
        return "DoS/DDoS"
    # WebAttack guess if HTTP(S) ports with abnormal sizes
    if dst_port in [80,443] and pd.notna(pps) and pps>200:
        return "WebAttack"
    # BruteForce guess: many small attempts forward
    if pd.notna(fwd) and pd.notna(bwd) and fwd>20 and bwd<5:
        return "BruteForce"
    return "Benign"

def recommendation_for(attack_type, risk_level):
    if attack_type == "BruteForce":
        return "Block source IP, enforce MFA, lock account after N failures, review auth logs."
    if attack_type == "PortScan":
        return "Apply firewall rules, block scanner IP, enable rate limits & port-knocking."
    if attack_type == "DoS/DDoS":
        return "Enable DDoS protection, throttle traffic, geo-block unusual regions, contact ISP."
    if attack_type == "WebAttack":
        return "Apply WAF rules, sanitize inputs, review server logs & patch vulnerable endpoints."
    if attack_type == "Exfiltration":
        return "Block outbound transfer, rotate keys, DLP policies, investigate compromised host."
    return "Monitor activity, whitelist trusted sources, tune alert thresholds."

def risk_from_labels(label):
    label = str(label).lower()
    if label in ["critical","attack","malicious"]: return "Critical"
    if label in ["high","highrisk","danger"]: return "High"
    if label in ["medium","med","suspicious"]: return "Medium"
    if label in ["low","benign","normal"]: return "Low"
    return "Medium"

# -------------------- Sidebar: Data Input --------------------
st.header("📥 Data Ingestion")
c1, c2 = st.columns(2)
df = None

with c1:
    up = st.file_uploader("Upload CSV (alerts/logs)", type=["csv"])
    if up:
        try:
            df = pd.read_csv(up)
            st.success(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} cols")
            st.dataframe(df.head(10), use_container_width=True)
            add_log("CSV uploaded.")
        except Exception as e:
            st.error(f"CSV read error: {e}")
            add_log(f"CSV error: {e}")

with c2:
    api = st.text_input("Or API endpoint (JSON records)")
    if st.button("Fetch from API"):
        try:
            r = requests.get(api, timeout=20)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data)
            st.success(f"API fetched: {len(df)} records")
            st.dataframe(df.head(10), use_container_width=True)
            add_log("API fetched.")
        except Exception as e:
            st.error(f"API error: {e}")
            add_log(f"API error: {e}")

if df is None:
    st.info("Upload CSV or fetch via API to continue.")
    st.stop()

# -------------------- Preprocess / Detect columns --------------------
st.divider()
st.header("🧹 Preprocessing & Column Detection")

time_col_name, time_series = to_datetime_smart(df)
src_col, dst_col = detect_ip_cols(df)
port_col = detect_port_col(df)

col_map = {"time": time_col_name, "src": src_col, "dst": dst_col, "port": port_col}

det1, det2, det3, det4 = st.columns(4)
det1.metric("Time Column", time_col_name or "Not found")
det2.metric("Source IP", src_col or "Not found")
det3.metric("Dest IP", dst_col or "Not found")
det4.metric("Port", port_col or "Not found")

# Manual overrides
with st.expander("🔧 Manual Column Overrides", expanded=False):
    col_map["time"] = st.selectbox("Timestamp column", [None]+list(df.columns), index=(0 if time_col_name is None else 1+list(df.columns).index(time_col_name)))
    col_map["src"]  = st.selectbox("Source IP column", [None]+list(df.columns), index=(0 if src_col is None else 1+list(df.columns).index(src_col)))
    col_map["dst"]  = st.selectbox("Destination IP column", [None]+list(df.columns), index=(0 if dst_col is None else 1+list(df.columns).index(dst_col)))
    col_map["port"] = st.selectbox("Port column", [None]+list(df.columns), index=(0 if port_col is None else 1+list(df.columns).index(port_col)))

# Parse timestamp if available
if col_map["time"]:
    df["_ts"] = pd.to_datetime(df[col_map["time"]], errors="coerce", utc=True)
else:
    df["_ts"] = pd.NaT

# Clean numeric matrix
X = clean_numeric(df)

st.success(f"Preprocessed numeric features: {X.shape[1]} columns")

# -------------------- Classification (Risk + Attack Type) --------------------
st.divider()
st.header("🧠 Risk Classification & Attack-Type Detection")

# Optionally map an existing label column to human risk
label_col = None
for c in df.columns:
    if "label" in c.lower() or "risk" in c.lower() or "attack" in c.lower():
        label_col = c; break

# Risk level (if label present), else default to Medium
if label_col:
    risk_level = df[label_col].apply(risk_from_labels)
else:
    risk_level = pd.Series(["Medium"]*len(df), index=df.index)

# Attack type (rules fallback)
attack_types = []
for _, row in df.iterrows():
    attack_types.append(rule_attack_type(row, col_map))
df["Predicted_Attack_Type"] = attack_types
df["Predicted_Risk_Level"]  = risk_level

# Recommendations
df["Recommendation"] = [recommendation_for(a, r) for a, r in zip(df["Predicted_Attack_Type"], df["Predicted_Risk_Level"])]

# Summary KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Alerts", len(df))
k2.metric("Critical/High", int(((df["Predicted_Risk_Level"]=="Critical") | (df["Predicted_Risk_Level"]=="High")).sum()))
k3.metric("Unique Source IPs", df[col_map["src"]].nunique() if col_map["src"] else 0)
k4.metric("Unique Dest Ports", df[col_map["port"]].nunique() if col_map["port"] else 0)

st.subheader("🔍 Classified Alerts (Sample)")
display_cols = ["Predicted_Risk_Level","Predicted_Attack_Type","Recommendation"]
for c in [col_map["src"], col_map["dst"], col_map["port"], label_col, col_map["time"]]:
    if c and c not in display_cols: display_cols.append(c)
st.dataframe(df[display_cols].head(300), use_container_width=True)

# -------------------- Anomaly Detection --------------------
st.divider()
st.header("🚨 Anomaly Detection (IsolationForest)")
with st.spinner("Training lightweight anomaly model on uploaded data..."):
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X)
    scores = iso.decision_function(X)
    preds  = iso.predict(X)  # -1 anomaly, 1 normal
df["AnomalyFlag"] = np.where(preds==-1, "Anomaly", "Normal")
anom_rate = (df["AnomalyFlag"]=="Anomaly").mean()*100

m1, m2 = st.columns(2)
m1.metric("Anomaly Rate", f"{anom_rate:.2f}%")
fig, ax = plt.subplots()
ax.hist(scores, bins=40)
ax.set_title("Anomaly Score Distribution")
ax.set_xlabel("Score"); ax.set_ylabel("Count")
m2.pyplot(fig)

# -------------------- Correlation & Patterns --------------------
st.divider()
st.header("🔗 Correlation & Patterns")

if col_map["time"] and df["_ts"].notna().any():
    per_min = df.set_index("_ts").resample("1T").size().rename("count").reset_index()
    st.subheader("Requests per Minute")
    st.line_chart(per_min.set_index("_ts")["count"])

if col_map["src"]:
    st.subheader("Top Source IPs")
    st.table(df[col_map["src"]].value_counts().head(15).rename_axis("Source IP").reset_index(name="Count"))

if col_map["dst"]:
    st.subheader("Top Destination IPs")
    st.table(df[col_map["dst"]].value_counts().head(15).rename_axis("Destination IP").reset_index(name="Count"))

if col_map["port"]:
    st.subheader("Top Destination Ports")
    st.table(df[col_map["port"]].value_counts().head(15).rename_axis("Port").reset_index(name="Count"))

# Day of week analysis
if col_map["time"] and df["_ts"].notna().any():
    tmp = df.copy()
    tmp["Day"] = tmp["_ts"].dt.tz_convert(None).dt.day_name()
    day_counts = tmp.groupby(["Day","Predicted_Attack_Type"]).size().unstack(fill_value=0)
    st.subheader("Attacks by Day of Week")
    st.dataframe(day_counts)
    fig2, ax2 = plt.subplots(figsize=(8,3))
    day_counts.plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

# -------------------- Export: Excel Report --------------------
st.divider()
st.header("📦 Report Generation & Export")

def build_excel_report(df, col_map):
    df2 = df.copy()

    # ✅ Remove timezone from ALL datetime columns automatically
    for col in df2.columns:
        try:
            converted = pd.to_datetime(df2[col], errors="ignore")
            if hasattr(converted, "dt"):
                try:
                    if converted.dt.tz is not None:
                        converted = converted.dt.tz_localize(None)
                except:
                    pass
                df2[col] = converted
        except:
            pass

    # ✅ Fix internal timestamp field
    if "_ts" in df2.columns:
        try:
            df2["_ts"] = pd.to_datetime(df2["_ts"], errors="coerce").dt.tz_localize(None)
        except:
            pass

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

        # Raw data
        df2.to_excel(writer, sheet_name="Raw_Alerts", index=False)

        # Summary
        summary = df2["Predicted_Risk_Level"].value_counts().rename_axis("Risk").reset_index(name="Count")
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # Attack Types
        atk = df2["Predicted_Attack_Type"].value_counts().rename_axis("AttackType").reset_index(name="Count")
        atk.to_excel(writer, sheet_name="Attack_Types", index=False)

        # Top IPs
        if col_map["src"]:
            df2[col_map["src"]].value_counts().head(50).rename_axis("SourceIP").reset_index(name="Count") \
                .to_excel(writer, sheet_name="Top_Source_IPs", index=False)

        # Top Ports
        if col_map["port"]:
            df2[col_map["port"]].value_counts().head(50).rename_axis("Port").reset_index(name="Count") \
                .to_excel(writer, sheet_name="Top_Ports", index=False)

        # Anomalies
        df2[df2["AnomalyFlag"] == "Anomaly"].to_excel(writer, sheet_name="Anomalies", index=False)

        # Pie chart
        wb = writer.book
        ws = writer.sheets["Summary"]
        chart = wb.add_chart({"type": "pie"})
        chart.add_series({
            "name": "Risk Distribution",
            "categories": ["Summary", 1, 0, len(summary), 0],
            "values": ["Summary", 1, 1, len(summary), 1],
        })
        chart.set_title({"name": "Risk Distribution"})
        ws.insert_chart("D2", chart)

    output.seek(0)
    return output


# ✅ IMPORTANT: This button must be here
try:
    report_bytes = build_excel_report(df, col_map)
    st.download_button(
        "⬇️ Download Excel Report",
        data=report_bytes,
        file_name="Security_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
except Exception as e:
    st.error(f"Report build error: {e}")                                                                                    # -------------------- Logs --------------------
st.divider() 
st.header("📝 Logs")
if not st.session_state.logs:
    st.info("No logs yet…")
else:
    for lg in st.session_state.logs[::-1]:
        st.write("•", lg)
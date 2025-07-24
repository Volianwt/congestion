# app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"
DATA_PATH = DERIVED / "vehicle_ci.parquet"

st.set_page_config(page_title="Urban Congestion Estimator", layout="wide")

@st.cache_data
def load_ci():
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    return df

df = load_ci()

st.title("Urban Congestion Estimator — MVP")

# ---- Sidebar filters ----
locs = sorted(df["location_name"].dropna().unique())
location = st.sidebar.selectbox("Location", locs, key="loc_select")

# 该地点的数据子集
loc_df = df[df["location_name"] == location]
if loc_df.empty:
    st.error("This location has no data at all.")
    st.stop()

# 这个地点的可用日期范围
date_min_loc = loc_df["hour"].min().date()
date_max_loc = loc_df["hour"].max().date()

# ---- Date range widget ----
# 用 location 做 key，换地点时会自动重置
date_value = st.sidebar.date_input(
    "Date range",
    value=(date_min_loc, date_max_loc),   # 默认：该地点的完整范围
    min_value=date_min_loc,
    max_value=date_max_loc,
    key=f"daterange_{location}",
)

# 兼容单日 / 范围
if isinstance(date_value, (list, tuple)):
    if len(date_value) == 2:
        start_date, end_date = date_value
    else:
        start_date = end_date = date_value[0]
else:
    start_date = end_date = date_value

# 保险起见再夹一下范围（避免用户手动回填越界）
if start_date < date_min_loc: start_date = date_min_loc
if end_date > date_max_loc:   end_date   = date_max_loc

# 转成 UTC Timestamp（半开区间）
start_ts = pd.Timestamp(start_date).tz_localize("UTC")
end_ts   = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)

mask = (
    (df["location_name"] == location) &
    (df["hour"] >= start_ts) &
    (df["hour"] <  end_ts)
)
sub = df.loc[mask].sort_values("hour")



st.subheader(location)
st.write(f"Rows: {len(sub)}")

if sub.empty:
    st.warning("No data for this range.")
    st.stop()

# y 轴选择
y_col = "volume_hour" if "volume_hour" in sub.columns else "volume_15min"

# ------------- 新版 Volume 图：一条灰线 + 彩色点 -------------
import plotly.graph_objects as go

# 颜色映射
color_map = {"low":"#1f77b4", "normal":"#2ca02c", "high":"#d62728", "unknown":"#7f7f7f"}

fig = go.Figure()

# 连续灰线（量值）
fig.add_trace(go.Scatter(
    x=sub["hour"], y=sub[y_col],
    mode="lines",
    name=y_col,
    line=dict(color="lightgray")
))

# 彩色点（CI 等级）
# --- 分等级添加散点，这样图例里会有 low/normal/high/unknown 四项 ---
for lvl, grp in sub.groupby("ci_level"):
    fig.add_trace(go.Scatter(
        x=grp["hour"], y=grp[y_col],
        mode="markers",
        name=f"{lvl}",                # 图例名字
        legendgroup="ci_level",       # 放在同一组
        marker=dict(size=7, color=color_map.get(lvl, "#7f7f7f")),
        text=[f"CI={v:.2f}" for v in grp["ci"]],
        hovertemplate="%{x}<br>Vol=%{y}<br>%{text}<extra></extra>",
        showlegend=True               # 确保显示
    ))


fig.update_layout(
    title=f"{y_col} (line) + CI level (colored dots)",
    xaxis_title="Time",
    yaxis_title="Volume"
)


fig.update_layout(legend_title_text="CI level / Volume")
st.plotly_chart(fig, use_container_width=True)
# -------------------------------------------------------------


# CI 自身曲线
# ------------- 新版 CI 图 -------------
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=sub["hour"], y=sub["ci"],
    mode="lines+markers",
    name="CI",
    marker=dict(size=6, color=[color_map[c] for c in sub["ci_level"]])
))
# 阈值线
fig2.add_hline(y=0.8, line_dash="dash", line_color="gray")
fig2.add_hline(y=1.2, line_dash="dash", line_color="gray")

fig2.update_layout(title="Congestion Index (CI)",
                   xaxis_title="Time", yaxis_title="CI")
fig2.update_layout(legend_title_text="CI level")
st.plotly_chart(fig2, use_container_width=True)
# --------------------------------------


with st.expander("Raw data preview"):
    st.dataframe(sub.head(200))

with st.expander("What does CI mean?"):
    st.markdown("""
    **CI = current hour’s volume ÷ historical mean (same location, weekday, hour).**  
    - <  0.8 → low  
    - 0.8 – 1.2 → normal  
    - \>  1.2 → high  
    - unknown → baseline missing/zero  
    """)

# -*- coding: utf-8 -*-
"""Phần tính toán của ứng dụng (không phụ thuộc Streamlit, có thể kiểm thử riêng).

Gồm: đọc và làm sạch tệp đơn hàng, dự báo nhu cầu (LightGBM học chung + naive),
lập kế hoạch cắt, khai phá combo (FP-Growth), phân tích giá trị đơn và mô phỏng chính sách.
"""
import itertools
import re
import unicodedata

import numpy as np
import pandas as pd

TOP_N = 30
TEST_WEEKS = 12


# ---------------------------------------------------------------- 1. ETL
def _clean_size(x):
    if x is None:
        return None
    x = x.strip().upper().replace("SIZE", "").strip().lstrip(":").strip().replace("FREESIZE", "FREE")
    x = re.split(r"\s", x)[0] if x else x
    return x or None


def parse_product(s):
    """Tách tên sản phẩm thành (style, màu, size)."""
    if s is None:
        return ("?", None, None)
    s = unicodedata.normalize("NFC", str(s)).strip()
    if s == "":
        return ("?", None, None)
    if s.upper().replace(" ", "") == "COD":
        return ("COD", None, None)
    m = re.search(r"^(.*?)\s*/\s*màu\s*:\s*(.*?)(?:\s*/\s*size\s*:\s*(.*?))?\s*$", s, flags=re.IGNORECASE)
    if m:
        return (m.group(1).strip().upper(), (m.group(2) or "").strip().upper() or None, _clean_size(m.group(3)))
    if s.count("/") >= 1:
        p = [x.strip() for x in s.split("/")]
        size = _clean_size(p[-1]) if len(p) >= 2 else None
        color = p[-2].strip().upper() if len(p) >= 3 else None
        nm = re.sub(r"^\d{6,}\s*-\s*", "", p[0])
        mc = re.search(r"\(([^)]+)\)", nm)
        code = mc.group(1).strip().upper() if mc else None
        nm = re.sub(r"^[^-]*-\s*\([^)]*\)\s*-?\s*", "", nm).strip()
        nm = re.sub(r"\(([^)]+)\)", "", nm).strip()
        return (code or (nm.upper() if nm else s.upper()), color, size)
    return (s.upper(), None, None)


def load_orders(file):
    """Đọc tệp Excel đơn hàng (xuất từ phần mềm bán hàng) và trả về bảng mỗi dòng = 1 sản phẩm."""
    df = pd.read_excel(file)
    need = ["STT", "Giá trị đơn hàng sau giảm giá", "Ngày tạo đơn", "Sản phẩm"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError("Tệp thiếu cột: " + ", ".join(miss))
    df = df[need].copy()
    df.columns = ["stt", "aov", "ngay", "sp"]
    df["is_head"] = pd.to_numeric(df["stt"], errors="coerce").notna()
    df["oid"] = df["is_head"].cumsum()
    df["ngay"] = df["ngay"].where(df["ngay"].astype(str).str.strip().ne("")).ffill()
    df["aov"] = pd.to_numeric(df["aov"], errors="coerce").where(df["is_head"]).ffill()
    df["dt"] = pd.to_datetime(df["ngay"], format="%d/%m/%Y", errors="coerce")
    d = df.dropna(subset=["sp"]).copy()
    d = d[d["sp"].astype(str).str.strip().ne("")]
    pr = d["sp"].map(parse_product)
    d["style"] = pr.map(lambda t: t[0])
    d["color"] = pr.map(lambda t: t[1])
    d["size"] = pr.map(lambda t: t[2])
    d = d.dropna(subset=["dt"])
    d["ds"] = (d["dt"] - pd.to_timedelta(d["dt"].dt.weekday, unit="D")).dt.normalize()
    return d[["oid", "dt", "ds", "aov", "style", "color", "size"]].reset_index(drop=True)


def summary(d):
    nit = d.groupby("oid").size()
    return {
        "Số dòng bán": int(len(d)),
        "Số đơn hàng": int(d["oid"].nunique()),
        "Số tuần có đơn": int(d["ds"].nunique()),
        "Số dòng sản phẩm": int(d["style"].nunique()),
        "Tỷ trọng COD (%)": float(round((d["style"] == "COD").mean() * 100, 1)),
        "Số món TB/đơn": float(round(nit.mean(), 2)),
        "Từ ngày": d["dt"].min().strftime("%d/%m/%Y"),
        "Đến ngày": d["dt"].max().strftime("%d/%m/%Y"),
    }


def weekly_total(d):
    weeks = pd.date_range(d["ds"].min(), d["ds"].max(), freq="W-MON")
    return d.groupby("ds").size().reindex(weeks, fill_value=0).rename("y")


# ---------------------------------------------------------------- 2. Dự báo
def build_panel(d, top_n=TOP_N):
    top = d["style"].value_counts().head(top_n).index.tolist()
    weekly = d[d["style"].isin(top)].groupby(["style", "ds"]).size().rename("y").reset_index()
    weeks = pd.date_range(d["ds"].min(), d["ds"].max(), freq="W-MON")
    panel = (pd.DataFrame(list(itertools.product(top, weeks)), columns=["unique_id", "ds"])
             .merge(weekly.rename(columns={"style": "unique_id"}), on=["unique_id", "ds"], how="left")
             .fillna({"y": 0}).sort_values(["unique_id", "ds"]).reset_index(drop=True))
    panel["y"] = panel["y"].astype(float)
    return panel


def _make_mlf():
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, num_leaves=15,
                              min_child_samples=20, subsample=0.8, verbosity=-1)
    return MLForecast(models={"LightGBM": model}, freq="W-MON", lags=[1, 2, 4, 8],
                      lag_transforms={1: [RollingMean(window_size=4), RollingMean(window_size=8)]},
                      date_features=["week", "month"])


def backtest(panel, test_weeks=TEST_WEEKS):
    """Kiểm định lùi cuốn chiếu: LightGBM so với naive và trung bình trượt 4 tuần."""
    cvm = _make_mlf().cross_validation(df=panel, h=1, step_size=1, n_windows=test_weeks).reset_index(drop=True)
    p = panel.sort_values(["unique_id", "ds"]).copy()
    g = p.groupby("unique_id")["y"]
    p["Naive"] = g.shift(1)
    p["TB trượt 4T"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    m = cvm[["unique_id", "ds", "y", "LightGBM"]].merge(p[["unique_id", "ds", "Naive", "TB trượt 4T"]],
                                                       on=["unique_id", "ds"])
    m["LightGBM"] = m["LightGBM"].clip(lower=0)
    return m


def scores(m):
    rows = []
    for c in ["LightGBM", "Naive", "TB trượt 4T"]:
        e = m["y"] - m[c]
        rows.append((c, np.sqrt((e ** 2).mean()), e.abs().mean(), e.abs().sum() / m["y"].sum() * 100))
    return pd.DataFrame(rows, columns=["Mô hình", "RMSE", "MAE", "WAPE (%)"]).round(2)


def dm_test(m, a="LightGBM", b="Naive"):
    from scipy import stats

    def _dm(dd):
        n = len(dd)
        s = dd.mean() / np.sqrt(dd.var(ddof=1) / n)
        return round(float(s), 3), round(float(2 * (1 - stats.t.cdf(abs(s), df=n - 1))), 4), n

    rows = []
    for name, loss in [("Sai số tuyệt đối", lambda e: e.abs()), ("Sai số bình phương", lambda e: e ** 2)]:
        dd = loss(m["y"] - m[a]) - loss(m["y"] - m[b])
        rows.append((name, "Từng ô", *_dm(dd)))
        rows.append((name, "Gộp theo tuần", *_dm(dd.groupby(m["ds"]).mean())))
    return pd.DataFrame(rows, columns=["Hàm tổn thất", "Cấp", "Thống kê DM", "p-value", "n"])


def forecast_next(panel):
    mlf = _make_mlf()
    mlf.fit(panel)
    fc = mlf.predict(h=1).rename(columns={"unique_id": "style", "LightGBM": "fc_style"})
    fc["fc_style"] = fc["fc_style"].clip(lower=0).round()
    return fc


def cut_plan(d, fc, recent_weeks=8):
    """Phân bổ dự báo cấp style xuống màu/size theo tỷ trọng 8 tuần gần nhất (phần dư lớn nhất)."""
    weeks = sorted(d["ds"].unique())
    start = weeks[-recent_weeks]
    sh = (d[(d["ds"] >= start) & (d["style"].isin(fc["style"]))]
          .groupby(["style", "color", "size"]).size().rename("n").reset_index())
    sh["share"] = sh["n"] / sh.groupby("style")["n"].transform("sum")
    plan = sh.merge(fc[["style", "fc_style"]], on="style")

    def alloc(g):
        total = int(g["fc_style"].iloc[0])
        raw = g["share"] * total
        q = np.floor(raw)
        rem = total - int(q.sum())
        idx = (raw - q).sort_values(ascending=False).index[:max(rem, 0)]
        q.loc[idx] += 1
        return q

    plan["cut_qty"] = plan.groupby("style", group_keys=False)[["share", "fc_style"]].apply(alloc).astype(int)
    plan = plan[plan["cut_qty"] > 0][["style", "color", "size", "cut_qty"]]
    return plan.sort_values(["cut_qty", "style"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------- 3. Combo
def baskets(d):
    bk = d[d["style"] != "COD"].groupby("oid")["style"].apply(lambda s: sorted(set(s)))
    return bk[bk.map(len) >= 1]


def combo_rules(d, min_support=0.002, min_conf=0.05, top=12):
    """FP-Growth (bản Python của mlxtend; cùng thuật toán với Spark MLlib trong notebook)."""
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    bk = baskets(d)
    te = TransactionEncoder()
    X = pd.DataFrame(te.fit(bk.tolist()).transform(bk.tolist()), columns=te.columns_)
    fi = fpgrowth(X, min_support=min_support, use_colnames=True)
    if fi.empty:
        return pd.DataFrame(columns=["A", "B", "support", "confidence", "lift", "so_don"]), len(bk)
    r = association_rules(fi, metric="confidence", min_threshold=min_conf)
    r = r[(r["antecedents"].map(len) == 1) & (r["consequents"].map(len) == 1)].copy()
    r["A"] = r["antecedents"].map(lambda s: next(iter(s)))
    r["B"] = r["consequents"].map(lambda s: next(iter(s)))
    r["pair"] = r.apply(lambda x: tuple(sorted([x["A"], x["B"]])), axis=1)
    r = r.sort_values(["lift", "confidence"], ascending=False).drop_duplicates("pair")
    out = r[["A", "B", "support", "confidence", "lift"]].head(top).round(4).reset_index(drop=True)
    out["so_don"] = (out["support"] * len(bk)).round().astype(int)
    return out, len(bk)


# ---------------------------------------------------------------- 4. Giá trị đơn
def aov_stats(d):
    ov = d.groupby("oid")["aov"].first().dropna()
    nit = d.groupby("oid").size().reindex(ov.index)
    return ov, nit


def simulate(ov, nit, conv_combo=0.15, disc=0.10, thr=300000, conv_thr=0.30):
    ppi = (ov / nit).median()
    base = ov.sum()
    n1 = int((nit == 1).sum())
    combo_up = n1 * conv_combo * ppi * (1 - disc)
    near = ov[(ov >= thr * 0.7) & (ov < thr)]
    thr_up = len(near) * (thr - near.mean()) * conv_thr if len(near) else 0.0
    return {
        "Giá mỗi món (trung vị)": ppi, "Số đơn 1 món": n1, "Số đơn gần ngưỡng": len(near),
        "Combo: doanh thu tăng (đ)": combo_up, "Combo: % tăng": combo_up / base * 100,
        "Ngưỡng: doanh thu tăng (đ)": thr_up, "Ngưỡng: % tăng": thr_up / base * 100,
    }

# -*- coding: utf-8 -*-
"""Ứng dụng Streamlit: hệ thống hỗ trợ quyết định cho Doanh nghiệp X.

Chạy:  streamlit run app.py
- Không tải tệp: xem các kết quả đã lưu trong thư mục ket_qua/ (không chứa dữ liệu khách hàng).
- Tải tệp đơn hàng (.xlsx): toàn bộ quy trình được tính lại trên dữ liệu mới.
"""
import os

import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Doanh nghiệp X – Big Data", page_icon="📊", layout="wide")
RES = os.path.join(os.path.dirname(__file__), "ket_qua")


def res(name):
    return os.path.join(RES, name)


def vnd(x):
    return f"{x:,.0f}".replace(",", ".") + " đ"


@st.cache_data(show_spinner="Đang đọc và làm sạch dữ liệu…")
def run_etl(file_bytes):
    import io
    return core.load_orders(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Đang chạy kiểm định lùi 12 tuần (khoảng 30 giây)…")
def run_forecast(d):
    panel = core.build_panel(d)
    m = core.backtest(panel)
    fc = core.forecast_next(panel)
    return panel, m, fc


@st.cache_data(show_spinner="Đang khai phá luật kết hợp…")
def run_combo(d, sup, conf):
    return core.combo_rules(d, min_support=sup, min_conf=conf)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Doanh nghiệp X")
st.sidebar.caption("Tiểu luận môn Nghiên cứu Dữ liệu lớn & Ứng dụng trong Kinh doanh – TS. Nguyễn Thôn Dã")
up = st.sidebar.file_uploader("Tải tệp đơn hàng (.xlsx)", type=["xlsx"])
st.sidebar.markdown(
    "Tệp cần có các cột: **STT, Ngày tạo đơn, Giá trị đơn hàng sau giảm giá, Sản phẩm**.  \n"
    "Dữ liệu chỉ xử lý trong phiên làm việc, không được lưu lại.")
LIVE = up is not None
if LIVE:
    d = run_etl(up.getvalue())
    st.sidebar.success("Đang dùng dữ liệu vừa tải lên")
else:
    st.sidebar.info("Chưa tải tệp: đang hiển thị kết quả đã lưu của tiểu luận")

st.title("Hệ thống hỗ trợ quyết định dựa trên dữ liệu")
st.caption("Dự báo nhu cầu · Kế hoạch cắt · Combo · Giá trị đơn hàng")

tabs = st.tabs(["Tổng quan", "Dự báo & kế hoạch cắt", "Combo", "Giá trị đơn", "Kiểm định & mở rộng"])

# ------------------------------------------------------------------ 1. Tổng quan
with tabs[0]:
    if LIVE:
        s = core.summary(d)
        cols = st.columns(4)
        for i, (k, v) in enumerate(s.items()):
            cols[i % 4].metric(k, f"{v:,}".replace(",", ".") if isinstance(v, int) else str(v).replace(".", ","))
        st.subheader("Doanh số theo tuần (số sản phẩm)")
        st.line_chart(core.weekly_total(d))
    else:
        cols = st.columns(4)
        for i, (k, v) in enumerate([("Số dòng bán", "17.642"), ("Số đơn hàng", "8.460"), ("Số tuần có đơn", "60 / 62"),
                                    ("Số dòng sản phẩm", "585"), ("Mã hàng chi tiết", "3.244"), ("Tỷ trọng COD", "10,9%"),
                                    ("Số món TB/đơn", "2,09"), ("Giai đoạn", "06/2025 – 07/2026")]):
            cols[i % 4].metric(k, v)
        st.info("Tải tệp đơn hàng ở thanh bên trái để tính lại toàn bộ trên dữ liệu mới.")

# ------------------------------------------------------------------ 2. Dự báo
with tabs[1]:
    if LIVE:
        panel, m, fc = run_forecast(d)
        st.subheader("Sai số trong kiểm định lùi 12 tuần (30 dòng bán chạy)")
        st.dataframe(core.scores(m), hide_index=True)
        sty = st.selectbox("Xem thực tế và dự báo của dòng sản phẩm", sorted(m["unique_id"].unique()))
        st.line_chart(m[m["unique_id"] == sty].set_index("ds")[["y", "LightGBM", "Naive"]]
                      .rename(columns={"y": "Thực tế"}))
        plan = core.cut_plan(d, fc)
        nxt = (panel["ds"].max() + pd.Timedelta(days=7)).strftime("%d/%m/%Y")
    else:
        st.subheader("So sánh 8 mô hình trong kiểm định lùi 12 tuần")
        st.dataframe(pd.read_excel(res("M1_ket_qua_sai_so.xlsx")), hide_index=True)
        if os.path.exists(res("Demo4_thuc_te_du_bao.png")):
            st.image(res("Demo4_thuc_te_du_bao.png"), caption="Thực tế và dự báo của hai dòng lớn nhất")
        plan = pd.read_excel(res("Demo3_ke_hoach_cat_khop_du_bao.xlsx"))
        nxt = "03/08/2026"
    st.subheader(f"Kế hoạch cắt tuần bắt đầu {nxt}")
    c1, c2 = st.columns([2, 1])
    c1.dataframe(plan.rename(columns={"style": "Dòng sản phẩm", "color": "Màu", "size": "Cỡ", "cut_qty": "Số lượng cắt"}),
                 hide_index=True, height=420)
    c2.metric("Tổng số lượng cắt", int(plan["cut_qty"].sum()))
    c2.metric("Số dòng sản phẩm", plan["style"].nunique())
    c2.download_button("Tải kế hoạch cắt (.csv)", plan.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"ke_hoach_cat_{nxt.replace('/', '-')}.csv", mime="text/csv")

# ------------------------------------------------------------------ 3. Combo
with tabs[2]:
    st.subheader("Cặp sản phẩm hay được mua cùng nhau (FP-Growth)")
    if LIVE:
        c1, c2 = st.columns(2)
        sup = c1.slider("Độ hỗ trợ tối thiểu (%)", 0.1, 1.0, 0.2, 0.05) / 100
        conf = c2.slider("Độ tin cậy tối thiểu (%)", 1, 30, 5) / 100
        rules, nb = run_combo(d, sup, conf)
        st.caption(f"Số giỏ hàng phân tích: {nb:,}".replace(",", "."))
    else:
        rules = pd.read_excel(res("M2_combo.xlsx"))
    show = rules.rename(columns={"A": "Mua A", "B": "Có B", "support": "Độ hỗ trợ", "confidence": "Độ tin cậy",
                                 "lift": "Độ nâng (lift)", "so_don": "Số đơn"})
    st.dataframe(show, hide_index=True)
    if len(rules):
        st.bar_chart(rules.assign(cap=rules["A"] + " → " + rules["B"]).set_index("cap")["lift"], horizontal=True)
    st.caption("Lift > 1: hai sản phẩm đi cùng nhau nhiều hơn ngẫu nhiên. Ưu tiên các cặp lift ≥ 2 để thử combo.")

# ------------------------------------------------------------------ 4. Giá trị đơn
with tabs[3]:
    if LIVE:
        ov, nit = core.aov_stats(d)
        c = st.columns(4)
        c[0].metric("AOV trung bình", vnd(ov.mean()))
        c[1].metric("AOV trung vị", vnd(ov.median()))
        c[2].metric("AOV đơn 1 món", vnd(ov[nit == 1].mean()))
        c[3].metric("AOV đơn nhiều món", vnd(ov[nit >= 2].mean()))
    else:
        m3 = pd.read_excel(res("M3_aov.xlsx")).set_index("Chỉ số")["Giá trị"]
        c = st.columns(4)
        c[0].metric("AOV trung bình", vnd(m3["AOV trung bình"]))
        c[1].metric("AOV trung vị", vnd(m3["AOV trung vị"]))
        c[2].metric("AOV đơn 1 món", vnd(m3["AOV đơn 1 món"]))
        c[3].metric("AOV đơn nhiều món", vnd(m3["AOV đơn nhiều món"]))
    st.subheader("Mô phỏng chính sách (kéo thanh trượt để đổi giả định)")
    a, b = st.columns(2)
    conv = a.slider("Combo: % đơn 1 món mua thêm", 5, 40, 15) / 100
    disc = a.slider("Combo: chiết khấu (%)", 0, 30, 10) / 100
    conv_t = b.slider("Freeship: % đơn gần ngưỡng mua thêm", 10, 60, 30) / 100
    if LIVE:
        thr = b.select_slider("Ngưỡng miễn phí vận chuyển (đ)", [250000, 300000, 350000, 400000], 300000)
        r = core.simulate(ov, nit, conv, disc, thr, conv_t)
        combo_pct, thr_pct = r["Combo: % tăng"], r["Ngưỡng: % tăng"]
        combo_vnd, thr_vnd = r["Combo: doanh thu tăng (đ)"], r["Ngưỡng: doanh thu tăng (đ)"]
    else:
        b.caption("Ngưỡng cố định 300.000 đ ở chế độ xem kết quả đã lưu.")
        base = 261084.33 * 8460
        combo_vnd = 3144 * conv * 120000 * (1 - disc)
        thr_vnd = 0.0164 * base * conv_t / 0.30
        combo_pct, thr_pct = combo_vnd / base * 100, thr_vnd / base * 100
    x, y, z = st.columns(3)
    x.metric("Combo", f"+{combo_pct:.2f}%", vnd(combo_vnd))
    y.metric("Ngưỡng freeship", f"+{thr_pct:.2f}%", vnd(thr_vnd))
    z.metric("Cộng gộp (ước lượng thô)", f"+{combo_pct + thr_pct:.2f}%")
    st.caption("Mô phỏng theo quy tắc, không phải ước lượng độ co giãn của cầu.")

# ------------------------------------------------------------------ 5. Kiểm định
with tabs[4]:
    st.subheader("Kiểm định Diebold–Mariano: LightGBM so với naive")
    if LIVE:
        st.dataframe(core.dm_test(m), hide_index=True)
    elif os.path.exists(res("Demo1_DM_test.xlsx")):
        st.dataframe(pd.read_excel(res("Demo1_DM_test.xlsx")), hide_index=True)
    st.caption("DM âm: LightGBM sai ít hơn naive. p-value < 0,05: khác biệt có ý nghĩa thống kê.")
    st.subheader("Khả năng mở rộng của Spark khi nhân dữ liệu lên 50 lần")
    if os.path.exists(res("Demo2_scalability.xlsx")):
        c1, c2 = st.columns([1, 1])
        c1.dataframe(pd.read_excel(res("Demo2_scalability.xlsx")), hide_index=True)
        if os.path.exists(res("Demo2_scalability.png")):
            c2.image(res("Demo2_scalability.png"))
    st.caption("Phép thử chạy bằng PySpark trong notebook (thư mục notebooks/).")

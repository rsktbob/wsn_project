"""產生 e01.drawio（D3QN／DQfD、Rainbow 兩頁）：經典 Dueling（Wang et al., 2016），4 段特徵依段號串成一個輸入。
風格沿用 docs/rl_si_pool_network.drawio：直式由上往下、頁寬 700、方塊高 56、輸入綠／網路紫／中間結果灰、
底部圖例、訓練說明與特徵表格。"""
from html import escape
from pathlib import Path

INPUT = "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#E1F5EE;strokeColor=#0F6E56;fontColor=#085041;fontSize=12;"
NET = "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#EEEDFE;strokeColor=#534AB7;fontColor=#3C3489;fontSize=12;"
MID = "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#F1EFE8;strokeColor=#888780;fontColor=#2C2C2A;fontSize=12;"
NOTE = "text;html=1;align=left;verticalAlign=top;fontSize=12;fontColor=#5F5E5A;whiteSpace=wrap;"
LEGEND_TEXT = "text;html=1;align=left;verticalAlign=middle;fontSize=12;fontColor=#5F5E5A;"
TABLE = "text;html=1;align=left;verticalAlign=top;fontSize=11;fontColor=#2C2C2A;whiteSpace=wrap;overflow=fill;"
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=classic;strokeColor=#5F5E5A;"
        "exitX={ex};exitY=1;exitDx=0;exitDy=0;entryX={nx};entryY=0;entryDx=0;entryDy=0;")

ROWS = [
    ("每段", "有效差異數分格", 4, "searcher、good 基因不同，且任一方解碼後有開的 sensor 數：0／1–2／3–4／≥5", "one-hot"),
    ("每段", "無效差異數", 1, "基因不同、但兩方都沒開的 sensor 數", "min(x, 5) / 5"),
    ("每段", "範圍長度", 1, "候選範圍的 sensor 數", "÷ 最長區段長度"),
    ("每段", "searcher 開啟比例", 1, "範圍內 searcher 有開的 sensor 比例", "0–1"),
    ("每段", "good 開啟比例", 1, "範圍內 good 有開的 sensor 比例", "0–1"),
    ("每段", "最低電量", 1, "範圍內 searcher 有開的 sensor 的最小剩餘電量", "E / E_max（沒開時 1）"),
    ("每段", "平均電量", 1, "同上，平均", "E / E_max"),
    ("每段", "瓶頸 sensor 數", 1, "searcher 解中 電量÷耗能 最低的 5 顆，落在範圍內的數量", "÷ 5"),
    ("每段", "耗能占比", 1, "範圍內 sensor 耗能 ÷ searcher 全網耗能", "0–1"),
    ("每段", "單一覆蓋 target 數", 1, "範圍內開啟 sensor 負責、且只被一顆 sensor 覆蓋的 target 數", "÷ 範圍內開啟數"),
    ("每段", "區段近期成功率", 1, "本 episode 內這段最近 50 次被選時有改善的比例", "0–1（沒紀錄 0.5）"),
    ("全域", "搜尋進度", 1, "已用評估 ÷ 預算", "0–1"),
    ("全域", "searcher 停滯", 1, "searcher 連續沒改善的回合數", "min(x, 20) / 20"),
    ("全域", "fitness 差距", 1, "(f_good − f_searcher) 在本回合 64 組交易中的排名", "0–1"),
    ("全域", "全網電量", 1, "活著的 sensor 平均剩餘電量", "E / E_max"),
]
TD = "border:1px solid #D3D1C7;padding:2px 5px;vertical-align:top;"
TH = TD + "background:#F1EFE8;font-weight:bold;"
FEATURE_TABLE = (
    '<table style="border-collapse:collapse;width:100%;">'
    f'<tr><th style="{TH}">類別</th><th style="{TH}">特徵</th><th style="{TH}">維度</th>'
    f'<th style="{TH}">怎麼算</th><th style="{TH}">正規化</th></tr>'
    + "".join(
        f'<tr><td style="{TD}">{g}</td><td style="{TD}">{n}</td><td style="{TD}text-align:center;">{d}</td>'
        f'<td style="{TD}">{how}</td><td style="{TD}">{norm}</td></tr>'
        for g, n, d, how, norm in ROWS
    )
    + f'<tr><td style="{TH}" colspan="2">合計</td><td style="{TH}text-align:center;">60</td>'
    f'<td style="{TH}" colspan="2">每段 14 × 4 段（依段號固定位置）＋ 全域 4</td></tr></table>'
)


def box(cid, label, x, y, w, style, h=56):
    return (f'<mxCell id="{cid}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(cid, src, dst, ex=0.5, nx=0.5):
    return (f'<mxCell id="{cid}" style="{EDGE.format(ex=ex, nx=nx)}" edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')


def page(pid, name, cells, height):
    return (f'<diagram id="{pid}" name="{escape(name)}"><mxGraphModel dx="900" dy="900" grid="1" gridSize="10" '
            f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="700" '
            f'pageHeight="{height}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            f'{"".join(cells)}</root></mxGraphModel></diagram>')


def top():
    """4 段特徵＋全域 → 串成 60 維 → 共用層 → 狀態表示（經典 Dueling：一次輸入整個狀態）。"""
    cells = []
    for k, x in enumerate([40, 200, 360, 520]):
        cells.append(box(f"seg{k}", f"<b>段 {k} 特徵 · 14 維</b><br>第 {k} 段候選範圍內", x, 20, 140, INPUT))
        cells.append(edge(f"es{k}", f"seg{k}", "cat", 0.5, round(0.1 + 0.2 * k, 2)))
    cells += [
        box("glob", "<b>全域特徵 · 4 維</b>（進度、停滯、fitness 差距、全網電量）", 40, 96, 620, INPUT, 36),
        box("cat", "<b>輸入 · 60 維（依段號固定位置串接）</b><br>段0 14 ‖ 段1 14 ‖ 段2 14 ‖ 段3 14 ‖ 全域 4",
            40, 164, 620, MID),
        box("enc", "<b>共用層</b><br>60 → 128 → 128，ReLU", 195, 260, 290, NET),
        box("h", "<b>狀態表示 · 128</b><br>一組交易一個（4 段的資訊都在裡面）", 195, 356, 290, MID),
        edge("eg", "glob", "cat", 0.5, 0.9), edge("ec", "cat", "enc"), edge("ee", "enc", "h"),
        edge("ev", "h", "V", 0.25, 0.5), edge("ea", "h", "A", 0.75, 0.5),
    ]
    return cells


def legend(y, params):
    return [
        box("lg0", "", 180, y + 6, 14, "rounded=1;html=1;fillColor=#E1F5EE;strokeColor=#0F6E56;", 12),
        box("lt0", "輸入", 200, y, 60, LEGEND_TEXT, 24),
        box("lg1", "", 270, y + 6, 14, "rounded=1;html=1;fillColor=#EEEDFE;strokeColor=#534AB7;", 12),
        box("lt1", f"有參數的網路（共 {params} 個參數）", 290, y, 300, LEGEND_TEXT, 24),
    ]


d3qn = top() + [
    box("V", "<b>V 分支 · 1 個數</b><br>128 → 64 → 1", 40, 464, 260, NET),
    box("A", "<b>A 分支 · 4 個數</b><br>128 → 64 → 4（第 k 個固定是第 k 段）", 340, 464, 300, NET),
    box("Q", "<b>Q(s,k) = V + A_k − mean A · 4 個數</b><br>V 對 4 段相同；段之間的差異來自 A", 100, 572, 480, MID),
    box("out", "<b>選 Q 最大的段</b><br>訓練 ε-greedy；拿去交換、突變", 190, 668, 300, MID),
    edge("evq", "V", "Q", 0.5, 0.25), edge("eaq", "A", "Q", 0.5, 0.75), edge("eqo", "Q", "out"),
] + legend(748, "41,157") + [
    box("train", "<b>訓練（同一個網路三種）</b>：D3QN＝Double DQN 1-step ＋ n-step（γ 0.5、n 3）＋ L2；"
        "<b>DQfD 式（主方法）</b>＝再加大邊界模仿 max_k[Q(s,k)+l(a_E,k)] − Q(s,a_E)（只算在作弊版示範上，margin 0.1），"
        "示範先預訓練、再永久留在 PER；純模仿＝只有大邊界損失<br>"
        "<b>參數</b>：共用層 7,808 ＋ 16,512、V 8,256 ＋ 65、A 8,256 ＋ 260 ＝ 41,157",
        40, 780, 620, NOTE, 100),
    box("tab", FEATURE_TABLE, 40, 890, 620, TABLE, 420),
]

rainbow = top() + [
    box("V", "<b>V 分支（Noisy）· 1 × 51</b><br>128 → 64 → 51", 40, 464, 260, NET),
    box("A", "<b>A 分支（Noisy）· 4 × 51</b><br>128 → 64 → 4×51（第 k 組固定是第 k 段）", 340, 464, 300, NET),
    box("dist", "<b>每段的回報分布 · 4 × 51</b><br>逐 atom：V + A_k − mean A，softmax；z ∈ [0, 2]", 100, 572, 480, MID),
    box("Q", "<b>Q(s,k) = Σ z_i · p_i(s,k) · 4 個數</b><br>段之間的差異來自 A", 190, 668, 300, MID),
    box("out", "<b>選 Q 最大的段</b><br>探索靠 Noisy 層（ε = 0）；拿去交換、突變", 190, 764, 300, MID),
    edge("evq", "V", "dist", 0.5, 0.25), edge("eaq", "A", "dist", 0.5, 0.75),
    edge("edq", "dist", "Q"), edge("eqo", "Q", "out"),
] + legend(844, "90,494") + [
    box("train", "<b>訓練</b>：Rainbow 六個元件全開（Double、Dueling、PER、n-step 3、Noisy、C51；γ 0.5）；"
        "<b>DQfD 式（主方法）</b>＝再加大邊界模仿（用期望 Q = Σ z·p 計算，只算在作弊版示範上），示範預訓練並永久留在 PER<br>"
        "<b>參數</b>：共用層 7,808 ＋ 16,512、V（Noisy）16,512 ＋ 6,630、A（Noisy）16,512 ＋ 26,520 ＝ 90,494",
        40, 876, 620, NOTE, 100),
    box("tab", FEATURE_TABLE, 40, 986, 620, TABLE, 420),
]

xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<mxfile host="app.diagrams.net">'
       + page("d3qn", "D3QN／DQfD 網路", d3qn, 1330) + page("rainbow", "Rainbow 網路", rainbow, 1430) + "</mxfile>\n")
Path(__file__).with_name("e01.drawio").write_text(xml, encoding="utf-8")
print("written")

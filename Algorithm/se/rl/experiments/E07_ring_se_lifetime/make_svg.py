"""產生 E07 簡單版兩張網路圖。"""
import sys
from pathlib import Path

OUT = Path(sys.argv[1])

HEAD = '''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{H}" viewBox="0 0 1000 {H}" font-family="Helvetica, Arial, 'Noto Sans CJK TC', sans-serif" font-size="12">
  <rect width="1000" height="{H}" fill="#ffffff"/>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#5F5E5A"/>
    </marker>
  </defs>
  <text x="500" y="32" text-anchor="middle" font-size="17" font-weight="bold" fill="#2C2C2A">{title}</text>

  <!-- ① 問題設定 -->
  <rect x="30" y="50" width="940" height="62" rx="8" fill="#F1EFE8" stroke="#888780"/>
  <text x="500" y="70" text-anchor="middle" font-weight="bold" fill="#2C2C2A">① 問題設定（同 E06）：原版 ring_se（n8 h4 w8 μ0.4，EN），每回合 64 組交易，每組在 4 個預抽候選範圍（80/20）裡選 1 個</text>
  <text x="500" y="87" text-anchor="middle" fill="#2C2C2A">網路替 4 個候選各打一個分數，選分數最高的；4 個候選用同一組權重（候選順序換了，結果跟著換）</text>
  <text x="500" y="104" text-anchor="middle" fill="#993C1D">預算 10000：原版 ring_se 1064.6（seed 7–36）；作弊版（偷看 4 候選只交換的 Δ）只多 +1.5%（1060.7 → 1076.2，seed 7–16）</text>
'''

def box(x, y, w, h, fill, stroke, lines, anchor_x=None):
    out = [f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>']
    cx = anchor_x or x + w / 2
    for i, (txt, size, bold, color) in enumerate(lines):
        yy = y + 19 + i * 17
        b = ' font-weight="bold"' if bold else ''
        out.append(f'  <text x="{cx:.0f}" y="{yy}" text-anchor="middle" font-size="{size}"{b} fill="{color}">{txt}</text>')
    return "\n".join(out)

def arrow(x1, y1, x2, y2):
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#arrow)"/>'

def example(y, lines):
    out = []
    for i, t in enumerate(lines):
        out.append(f'  <text x="700" y="{y + i * 15}" font-size="11" fill="#5F5E5A">{t}</text>')
    return "\n".join(out)

P, PS = "#3C3489", "#534AB7"
G, GS = "#085041", "#0F6E56"

def lane_box(x, y, w, h, fill, stroke, color, lines):
    out = [f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>']
    for i, (txt, bold) in enumerate(lines):
        b = ' font-weight="bold"' if bold else ''
        out.append(f'  <text x="{x + w / 2:.0f}" y="{y + 17 + i * 15}" text-anchor="middle" font-size="11"{b} fill="{color}">{txt}</text>')
    return "\n".join(out)


def network(top, params, dqn):
    s = []
    s.append(f'  <text x="40" y="{top + 22}" font-size="14" font-weight="bold" fill="{PS}">③ 網路（{params}）：4 個候選各走一條路，4 條路是<tspan fill="#993C1D">同一個網路、同一組權重</tspan>，最後才放在一起比</text>')
    y = top + 36
    s.append(f'  <text x="40" y="{y + 16}" font-size="12" font-weight="bold" fill="{G}">共用輸入</text>')
    s.append(f'  <text x="40" y="{y + 31}" font-size="11" fill="#5F5E5A">4 條路都會用到</text>')
    s.append(box(180, y, 780, 40, "#E1F5EE", GS, [("候選特徵 4×10（每個候選一列：位置 4、範圍外影響 4、瓶頸 2）　‖　全域 10（4 個候選用同一份）", 11, True, G)]))
    y += 52
    LX = [180 + i * 196 for i in range(4)]
    W = 184
    L = [3, 25, 12, 1]
    rows = [
        ("輸入", "範圍內的 sensor", "#E1F5EE", GS, G,
         lambda i: [(f"候選 {i + 1}：範圍內 sensor", True), (f"L{i + 1} × 25（例 L{i + 1} ＝ {L[i]}）", False)]),
        ("第 1 步", "整理每顆 sensor", "#EEEDFE", PS, P,
         lambda i: [("每顆 25 → 64 → 64", True), (f"→ L{i + 1} × 64", False)]),
        ("第 2 步", "合成一個候選", "#EEEDFE", PS, P,
         lambda i: [("平均 ‖ 最大", True), ("→ 128（不管 L 多少）", False)]),
        ("第 3 步", "加上候選特徵、全域", "#EEEDFE", PS, P,
         lambda i: [(f"‖ 候選 {i + 1} 特徵 10 ‖ 全域 10", True), (f"148→128→64 ＝ u{i + 1}", False)]),
    ]
    if dqn:
        rows.append(("第 4 步", "每個候選的優勢", "#FAEEDA", "#BA7517", "#633806",
                     lambda i: [(f"u{i + 1} → 1 ＝ A{i + 1}", True), ("比平均好多少", False)]))
    else:
        rows.append(("第 4 步", "每個候選一個分數", "#FBEAF0", "#993556", "#72243E",
                     lambda i: [(f"u{i + 1} → 1 ＝ 分數 s{i + 1}", True), ("越高越該選", False)]))
    for r, (label, sub, fill, stroke, color, fn) in enumerate(rows):
        s.append(f'  <text x="40" y="{y + 16}" font-size="12" font-weight="bold" fill="{color}">{label}</text>')
        s.append(f'  <text x="40" y="{y + 31}" font-size="11" fill="#5F5E5A">{sub}</text>')
        for i, x in enumerate(LX):
            s.append(lane_box(x, y, W, 38, fill, stroke, color, fn(i)))
            if r < len(rows) - 1:
                s.append(arrow(x + W / 2, y + 38, x + W / 2, y + 52))
        y += 52
    # 匯合
    for x in LX:
        s.append(arrow(x + W / 2, y - 14, x + W / 2, y + 6))
    y += 8
    if dqn:
        s.append(box(180, y, 780, 44, "#FAEEDA", "#BA7517", [("V ＝ 平均(u1, u2, u3, u4) → 64 → 1：這組交易整體能改善多少（4 個候選共用一個 V）", 11, True, "#633806"), ("V 用第 3 步的 u 算；A1…A4 直接往下傳到第 6 步", 10, False, "#633806")]))
        s.append(f'  <text x="40" y="{y + 16}" font-size="12" font-weight="bold" fill="#633806">第 5 步</text>')
        s.append(f'  <text x="40" y="{y + 31}" font-size="11" fill="#5F5E5A">整體價值</text>')
        y += 44
        s.append(arrow(570, y, 570, y + 14)); y += 16
        s.append(box(180, y, 780, 44, "#534AB7", PS, [("Qc ＝ V ＋ Ac − 平均(A1…A4)　→　Q1、Q2、Q3、Q4", 12, True, "#ffffff"), ("Q ＝ 選這個候選的預期獎勵；選 Q 最大，訓練時 ε-greedy", 11, False, "#ffffff")]))
        s.append(f'  <text x="40" y="{y + 18}" font-size="12" font-weight="bold" fill="{PS}">第 6 步</text>')
        s.append(f'  <text x="40" y="{y + 33}" font-size="11" fill="#5F5E5A">4 個 Q 一起比</text>')
        y += 44
        ex = ["例：A＝0.2、1.0、−0.8、−0.4（平均 0），V＝0.5 → Q＝0.7、1.5、−0.3、0.1 → 選候選 2"]
    else:
        s.append(box(180, y, 780, 44, "#534AB7", PS, [("softmax(s1, s2, s3, s4) → 機率 π1、π2、π3、π4（加起來 = 1）", 12, True, "#ffffff"), ("選機率最大的候選", 11, False, "#ffffff")]))
        s.append(f'  <text x="40" y="{y + 18}" font-size="12" font-weight="bold" fill="{PS}">第 5 步</text>')
        s.append(f'  <text x="40" y="{y + 33}" font-size="11" fill="#5F5E5A">4 個一起比</text>')
        y += 44
        ex = ["例：分數 s＝0.5、2.0、−1.0、0.0 → 機率 16%、71%、4%、10% → 選候選 2"]
    for t in ex:
        s.append(f'  <text x="570" y="{y + 18}" text-anchor="middle" font-size="11" fill="#5F5E5A">{t}</text>')
    y += 26
    note = "和 E06 比：拿掉 sensor Transformer、範圍外 sensor、候選間 attention。E06 拿掉 Transformer（0 層）945.7 對 944.6，沒差"
    s.append(f'  <text x="40" y="{y + 14}" font-size="11" fill="#993C1D">{note}</text>')
    y += 18
    s.insert(0, f'  <rect x="20" y="{top}" width="960" height="{y + 14 - top}" rx="12" fill="none" stroke="{PS}" stroke-dasharray="7 4"/>')
    return "\n".join(s), y + 22


def supervised():
    H = 1000
    out = [HEAD.format(H=H, title="E07 監督式（簡單版）：學作弊版選 4 個候選中的哪一個")]
    out.append('''
  <!-- ② 標籤資料 -->
  <rect x="30" y="124" width="940" height="100" rx="8" fill="#FAEEDA" stroke="#BA7517"/>
  <text x="50" y="146" font-size="14" font-weight="bold" fill="#633806">② 標籤資料（collect.py 全部重新蒐集，不用舊資料）</text>
  <g fill="#633806">
    <text x="50" y="168"><tspan font-weight="bold">RLTRAIN01–30</tspan>：每張從滿電用<tspan font-weight="bold">原版 ring_se 推進</tspan>（4 選 1 隨機），<tspan font-weight="bold">每段預算 10000</tspan>（約 78 回合），跑完整 lifetime（約 34 段）</text>
    <text x="50" y="186">每 3 段標一段、起點依地圖錯開；<tspan font-weight="bold">作弊版只負責標答案</tspan>：偷看 4 個候選「只交換」的結果（不計預算、不用演算法亂數 → 推進和原版完全相同）</text>
    <text x="50" y="204"><tspan font-weight="bold">標籤 ＝ 作弊版會選的候選</tspan>：一般 Δ ＝ max(0, f(c1) − f(S)) ＋ max(0, f(c2) − f(G)) 最大的，同分再比 f(c2)；每組交易都有答案</text>
    <text x="50" y="220" font-size="11">預估約 180 萬組交易、約 1.5 GB、8 shard 約 1 小時；不另切驗證圖</text>
  </g>
''')
    net, y = network(238, "約 3.3 萬參數；E06 是 15.4 萬", dqn=False)
    out.append(net)
    y += 8
    out.append(f'''
  <!-- ④ 訓練 -->
  <rect x="30" y="{y}" width="940" height="128" rx="10" fill="#FBEAF0" stroke="#993556"/>
  <text x="50" y="{y + 24}" font-size="14" font-weight="bold" fill="#72243E">④ 訓練（train_sup.py）</text>
  <g fill="#72243E">
    <text x="50" y="{y + 50}"><tspan font-weight="bold">損失</tspan>　交叉熵 −log π_(作弊版選的候選)，全部交易都用</text>
    <text x="50" y="{y + 68}" font-size="11">　　　　例：作弊版選候選 2，網路給它 71% → 損失 −log 0.71 ＝ 0.34；若作弊版選候選 3（網路只給 4%）→ 損失 3.2，候選 3 的分數會被往上調</text>
    <text x="50" y="{y + 92}"><tspan font-weight="bold">設定</tspan>　Adam 3e-4（cosine 衰減）、L2 1e-5、batch 256、固定 5 個 epoch、不早停；評估用最後一個 checkpoint</text>
    <text x="50" y="{y + 112}"><tspan font-weight="bold">記錄</tspan>　每 epoch 訓練集交叉熵、和作弊版一致的比例（隨機 25%）、4 個位置被選比例</text>
  </g>''')
    y += 140
    out.append(f'''
  <!-- ⑤ 評估 -->
  <rect x="30" y="{y}" width="940" height="62" rx="8" fill="#E1F5EE" stroke="#0F6E56"/>
  <text x="500" y="{y + 20}" text-anchor="middle" font-weight="bold" fill="#085041">⑤ 評估：MAP01–06 × seed 7–36（180 場），預算 10000，成對 Wilcoxon 對 ring_se 1064.6、E07 D3QN</text>
  <text x="500" y="{y + 38}" text-anchor="middle" fill="#085041">時間：蒐集約 1 小時 → 訓練約 20 分 → 評估約 2.5–3 小時（ring_se 10000 每場約 5 分，加上網路推論）</text>
  <text x="500" y="{y + 54}" text-anchor="middle" font-size="11" fill="#993C1D">天花板很低：作弊版只 +1.5%，網路拿到一半約 +0.8%（約 8 單位）</text>
</svg>''')
    y += 74
    text = "\n".join(out).replace(f'height="{H}"', f'height="{y}"').replace(f'0 0 1000 {H}', f'0 0 1000 {y}')
    (OUT / "supervised" / "network.svg").write_text(text, encoding="utf-8")

def d3qn():
    H = 1000
    out = [HEAD.format(H=H, title="E07 D3QN（簡單版）：同樣的候選編碼＋Dueling，30 張圖完整 lifetime、每段預算 1000")]
    out.append('''
  <!-- ② 環境 -->
  <rect x="30" y="124" width="940" height="100" rx="8" fill="#E6F1FB" stroke="#185FA5"/>
  <text x="50" y="146" font-size="14" font-weight="bold" fill="#0C447C">② 環境與獎勵（不偷看，網路自己推進）</text>
  <g fill="#0C447C">
    <text x="50" y="168"><tspan font-weight="bold">RLTRAIN01–30</tspan>（同監督式）每張從滿電跑完整 lifetime，<tspan font-weight="bold">每段預算 1000</tspan>（使用者 10/19 從 10000 改）；一段最佳化（約 8 回合）＝一個 episode</text>
    <text x="50" y="186">一張圖約 32 個 episode，<tspan font-weight="bold">只跑 1 輪（約 950 個 episode）</tspan>，地圖順序打亂</text>
    <text x="50" y="204"><tspan font-weight="bold">獎勵</tspan>：同 E06 差異獎勵，用實際子代（含突變）；÷ 上一回合一般 Δ 的正值中位數、上限 5</text>
    <text x="50" y="220" font-size="11">差異獎勵：同一個 searcher 的 8 組只會收最好的 child1 → 只有真的被收下的那組拿到「比其他組多出來的部分」</text>
  </g>
''')
    net, y = network(238, "約 3.7 萬參數；E06 是 15.4 萬", dqn=True)
    out.append(net)
    y += 8
    out.append(f'''
  <!-- ④ 訓練 -->
  <rect x="30" y="{y}" width="940" height="150" rx="10" fill="#E6F1FB" stroke="#185FA5"/>
  <text x="50" y="{y + 24}" font-size="14" font-weight="bold" fill="#0C447C">④ 訓練（train_rl.py）</text>
  <g fill="#0C447C">
    <text x="50" y="{y + 50}"><tspan font-weight="bold">目標</tspan>　γ = 0：讓 Q_(選的候選) 接近實際拿到的獎勵 r（Huber）；Dueling、PER（α 0.4、β 0.6→1）</text>
    <text x="50" y="{y + 68}" font-size="11">　　　　例：選了候選 2、Q2＝1.5，實際獎勵 r＝0.8 → 把 Q2 往 0.8 拉（只更新選到的那個）；γ = 0 時目標就是 r，不需要 target network、Double DQN（程式沒有實作）</text>
    <text x="50" y="{y + 92}"><tspan font-weight="bold">Replay</tspan>　FIFO 上限 15,000 回合（預算 1000 時全部只約 7,600 回合 → 等於全留，同 E06），PER 抽樣在 GPU 上做</text>
    <text x="50" y="{y + 114}"><tspan font-weight="bold">設定</tspan>　從零、ε 1.0 → 0.05（前 50% episode）、warmup 5,000 組、每回合更新 16 次（同 E06；共約 12 萬次）、batch 128、lr 1e-4、L2 1e-5</text>
    <text x="50" y="{y + 134}"><tspan font-weight="bold">驗證</tspan>　訓練中不驗證；每 100 episode 存 checkpoint，評估用最後一個；記錄每張訓練圖的 lifetime 當學習曲線</text>
  </g>''')
    y += 162
    out.append(f'''
  <!-- ⑤ 評估 -->
  <rect x="30" y="{y}" width="940" height="62" rx="8" fill="#E1F5EE" stroke="#0F6E56"/>
  <text x="500" y="{y + 20}" text-anchor="middle" font-weight="bold" fill="#085041">⑤ 評估：MAP01–06 × seed 7–11（30 場），預算 1000，成對 Wilcoxon 對同場 ring_se 899.9</text>
  <text x="500" y="{y + 38}" text-anchor="middle" fill="#085041">對照：E07 監督式（差異獎勵、預算 1000 標籤）947.6 · E06 監督式 950.3 · 作弊版 982.1（同場 seed 7–11）</text>
  <text x="500" y="{y + 54}" text-anchor="middle" font-size="11" fill="#993C1D">時間：訓練約 30 分 → 評估約 5 分；E06 從零 RL 比監督式低約 1%</text>
</svg>''')
    y += 74
    text = "\n".join(out).replace(f'height="{H}"', f'height="{y}"').replace(f'0 0 1000 {H}', f'0 0 1000 {y}')
    text = text.replace("預算 10000：原版 ring_se 1064.6（seed 7–36）；作弊版（偷看 4 候選只交換的 Δ）只多 +1.5%（1060.7 → 1076.2，seed 7–16）",
                        "預算 1000（seed 7–11）：原版 ring_se 899.9；作弊版（偷看 4 候選只交換的 Δ）982.1（+9.1%）")
    (OUT / "d3qn" / "network.svg").write_text(text, encoding="utf-8")

supervised()
d3qn()

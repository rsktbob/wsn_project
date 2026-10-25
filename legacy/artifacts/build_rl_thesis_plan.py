from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = OUTPUT_DIR / "RL導引超啟發式WSN研究與論文時程規劃.docx"

FONT_NAME = "Microsoft JhengHei"
NAVY = "1F4E78"
PALE_BLUE = "EEF4F8"
LIGHT_BLUE = "D9EAF7"
LIGHT_GRAY = "D9D9D9"
MID_GRAY = "666666"
BLACK = "000000"
WHITE = "FFFFFF"


def set_run_font(run, size=None, bold=None, color=BLACK, italic=None):
    run.font.name = FONT_NAME
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT_NAME)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT_NAME)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT_NAME)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=75, start=90, bottom=75, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color=LIGHT_GRAY, size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def keep_row_together(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    tr_pr.append(cant_split)


def set_repeat_keep(paragraph, keep_with_next=True, keep_together=False):
    p_pr = paragraph._p.get_or_add_pPr()
    if keep_with_next:
        node = OxmlElement("w:keepNext")
        p_pr.append(node)
    if keep_together:
        node = OxmlElement("w:keepLines")
        p_pr.append(node)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_run_font(run, 9, color=MID_GRAY)
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    run2 = paragraph.add_run(" 頁")
    set_run_font(run2, 9, color=MID_GRAY)


def set_paragraph_text(paragraph, text, *, size=10.5, bold=False, color=BLACK, align=None):
    paragraph.clear()
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.08
    return paragraph


def add_body(doc, text, *, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.25
    if bold_lead and text.startswith(bold_lead):
        lead = p.add_run(bold_lead)
        set_run_font(lead, 11, bold=True)
        rest = p.add_run(text[len(bold_lead):])
        set_run_font(rest, 11)
    else:
        run = p.add_run(text)
        set_run_font(run, 11)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.28 + level * 0.22)
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    marker = "" if text.startswith("□") else "•  "
    run = p.add_run(marker + text)
    set_run_font(run, 10.5)
    return p


def add_numbered(doc, text, number):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.first_line_indent = Inches(-0.20)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(f"{number}.  {text}")
    set_run_font(run, 10.5)
    return p


def remove_paragraph_border(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    existing = p_pr.find(qn("w:pBdr"))
    if existing is not None:
        p_pr.remove(existing)
    p_bdr = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right", "between"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "nil")
        p_bdr.append(node)
    p_pr.append(p_bdr)


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    set_repeat_keep(p, keep_with_next=True)
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    for run in p.runs:
        set_run_font(run, 17 if level == 1 else 13, bold=True, color=BLACK)
    return p


def start_new_page(doc):
    """Start following content on a new page without creating an empty page.

    A page-break character placed after a full table can be pushed onto the next
    page and create an extra blank page. A paragraph with page_break_before does
    not have that problem.
    """
    p = doc.add_paragraph()
    p.paragraph_format.page_break_before = True
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 0.1
    run = p.add_run(" ")
    set_run_font(run, 1, color=WHITE)
    return p


def add_table(doc, headers, widths, rows, *, font_size=9.5, alignments=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    repeat_table_header(header)
    for i, (cell, heading, width) in enumerate(zip(header.cells, headers, widths)):
        cell.width = Inches(width)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(cell, NAVY)
        set_cell_margins(cell, top=95, bottom=95)
        set_paragraph_text(cell.paragraphs[0], heading, size=9.5, bold=True, color=WHITE,
                           align=WD_ALIGN_PARAGRAPH.CENTER)
    for row_index, values in enumerate(rows):
        row = table.add_row()
        keep_row_together(row)
        for col_index, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            if row_index % 2 == 1:
                set_cell_shading(cell, PALE_BLUE)
            align = WD_ALIGN_PARAGRAPH.LEFT
            if alignments and alignments[col_index] == "center":
                align = WD_ALIGN_PARAGRAPH.CENTER
            p = cell.paragraphs[0]
            set_paragraph_text(p, str(value), size=font_size, align=align)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(2)
    return table


WEEK_CONTENT = [
    ("題目與現況盤點", "1. 盤點目前可重現的 SA SETS、GI GOMEA、Target 與 Sensor encoding、fitness service、連續感測半徑及 RLHH 程式。\n2. 建立版本與實驗清單，標示可用、待驗證、暫停三種狀態。\n3. 向系所確認口試申請、論文繳交與離校行政規則。", "第 1 章列出研究背景與目前問題。\n產出：研究現況一頁摘要、期限確認表、程式與資料盤點表。"),
    ("題目與研究問題", "1. 擬定三個題目版本：RL 選擇搜尋算子、RL 切換探索與局部搜尋階段、RL 導引移動。\n2. 依訓練樣本數、可驗證性、實作風險與論文貢獻比較。\n3. 優先保留 RL 導引搜尋算子或搜尋階段，移動控制列為備案。", "第 1 章完成研究動機初稿 800 至 1200 字。\n產出：題目決策矩陣、暫定研究問題 RQ1 至 RQ5。"),
    ("文獻檢索", "1. 建立 2021 至 2026 年文獻檢索式與納入排除規則。\n2. 分類 WSN 壽命最佳化、超啟發式、超啟發式控制器、RL 導引最佳化、連續感測半徑。\n3. 建立文獻表，先收錄 40 篇候選並精讀至少 12 篇。", "第 2 章完成章節架構與方法分類表。\n產出：文獻資料庫、檢索紀錄、每篇 150 至 250 字摘要。"),
    ("題目定案", "1. 與指導教授確認主要貢獻只聚焦一項：RL 導引超啟發式搜尋。\n2. 決定連續感測半徑是問題模型改良、次要貢獻或僅作敏感度分析。\n3. 鎖定研究問題、比較基準、主要指標與不可超出的範圍。", "第 1 章完成 0.3 版。\n產出：一頁研究契約，含題目、問題、假設、貢獻、排除項目。\n里程碑：2026/10/9 題目範圍凍結。"),
    ("系統模型", "1. 逐項核對節點、目標、基地台、能量、感測、覆蓋、連通、路由、排程與移動模型。\n2. 為 state、decoder、problem、routing service、fitness service 畫資料流程。\n3. 列出離散與連續感測模式的共同介面及差異。", "第 3 章完成 3.1 至 3.3 初稿。\n產出：系統模型圖、符號表、程式與公式對照表。"),
    ("基準線重現", "1. 固定環境、亂數種子、執行預算與輸出欄位。\n2. 以 5 個種子重跑目前 SA SETS 主版本。\n3. 確認每次實驗記錄 decoder、mutation、fitness service、sensing mode、程式版本與硬體。", "第 3 章完成目標函數與限制條件草稿。\n產出：可重跑指令、基準 CSV、設定 JSON、失敗原因紀錄。"),
    ("基準演算法", "1. 以相同預算重跑 GI GOMEA 與目前最強的 SA SETS 變體。\n2. 排除不可公平比較或無法穩定執行的版本，保留原因。\n3. 檢查 lifetime、耗能、覆蓋、連通、評估次數與時間的計算一致性。", "第 2 章撰寫超啟發式與 GOMEA 類方法。\n產出：基準比較表、演算法公平性檢查表。"),
    ("基準線凍結", "1. 建立統一 experiment runner 與結果彙整流程。\n2. 對同一解重複評估，檢查 fitness、cost 與 routing 是否一致。\n3. 決定正式比較的 3 至 5 個基準演算法。", "第 2 章撰寫 RL 導引最佳化與 hyper heuristic。\n產出：測試報告、基準清單。\n里程碑：2026/11/6 基準線介面凍結。"),
    ("RL 問題定義", "1. 定義固定長度 state：進度、停滯、近期改善率、可行性缺口、能量平衡、解多樣性與前一動作。\n2. 定義 action：選擇現有搜尋算子或切換探索、利用與修復階段。\n3. 定義 reward：正規化改善量扣除評估成本與不可行懲罰。", "第 4 章完成方法總覽與 MDP 定義。\n產出：state action reward 規格表、每個特徵的數值範圍。"),
    ("RL 最小版本", "1. 先以 tabular Q learning 或小型離散控制器建立可解釋基準。\n2. 用既有搜尋紀錄估計 state 分布、reward 尺度與每次求解可產生的 transition 數。\n3. 若 transition 太少，改採 contextual bandit 或跨多個實例共同訓練。", "第 4 章撰寫控制器、更新規則與訓練資料來源。\n產出：最小可執行控制器、訊號分布圖。"),
    ("整合與測試", "1. 建立 RL controller 與 metaheuristic 之間的 adapter，不改變原搜尋算子的內部行為。\n2. 加入 deterministic 測試、reward 單元測試、動作合法性與模型儲存測試。\n3. 確認關閉 RL 時能完全回復固定策略基準。", "第 4 章完成演算法流程與虛擬碼初稿。\n產出：整合測試、對照模式、實作說明。"),
    ("第一次先導實驗", "1. 在單一小型案例以 5 個相同種子比較固定 SA SETS 與 RL 導引版本。\n2. 觀察 reward、action 使用率、停滯長度、fitness 改善與額外時間。\n3. 只修正明確 bug，不依單一結果反覆調參。", "第 3 章完成第一版完整草稿。\n產出：先導實驗報告、問題清單。\n里程碑：2026/12/4 最小方法可執行。"),
    ("學習訊號診斷", "1. 檢查 action 是否塌縮為單一選項，reward 是否被不可行懲罰主導。\n2. 比較短期 reward 與最終 lifetime 的相關性。\n3. 若 credit assignment 太延遲，加入固定視窗改善或階段結束回饋。", "第 4 章補充設計理由與失敗條件。\n產出：訊號診斷圖、reward 版本決策紀錄。"),
    ("控制器改良", "1. 完成 state 正規化、epsilon 排程、重播或更新頻率設計。\n2. 限制 RL 額外成本，預設每固定評估次數才做一次決策。\n3. 執行 10 個種子的小型對照並檢查結果方向。", "第 4 章完成狀態、動作、獎勵與訓練流程 0.6 版。\n產出：第二次先導結果與 overhead 分析。"),
    ("範圍取捨", "1. 比較離散與連續感測是否會改變 RL 結論。\n2. 距離感知 fitness 僅在有明確假設與消融設計時納入；否則凍結於現行 fitness。\n3. 刪除無法在期限內形成獨立證據的支線。", "第 2 章完成第一版完整草稿。\n產出：研究範圍刪減清單、感測與 fitness 的實驗角色。"),
    ("重現性緩衝", "1. 整理設定檔、輸出格式、checkpoint 與異常中止續跑。\n2. 建立自動摘要，產生每 seed lifetime、時間、評估次數與失敗類型。\n3. 補齊 2026 年底前所有單元與 smoke tests。", "整理參考文獻格式、圖表命名規則與第 5 章結果表模板。\n產出：可交接的實驗 README、資料字典。"),
    ("跨案例先導", "1. 選小、中、大三種網路規模，各跑 10 個配對種子。\n2. 檢查 state 尺度是否跨案例可比，必要時只改正規化方式。\n3. 記錄訓練與測試案例，禁止測試資料洩漏。", "第 4 章完成 0.7 版，補上複雜度與實作細節。\n產出：跨規模結果與泛化問題清單。"),
    ("方法凍結", "1. 與指導教授檢視先導結果、action 行為與研究貢獻。\n2. 決定最終 RL 類型、state 特徵、action 集合與 reward。\n3. 先列出正式消融，不再依結果新增任意版本。", "第 4 章完成 0.8 版。\n產出：方法規格 1.0、消融清單、正式實驗准入判斷。\n里程碑：2027/1/15 方法凍結。"),
    ("實驗設計", "1. 定義案例因素：規模、sensor target 密度、初始能量異質性、拓樸、感測模式與移動情境。\n2. 定義主指標與次指標。\n3. 正式實驗原則採相同種子、相同停止條件與相同計算預算。", "第 5 章完成 5.1 實驗目的、5.2 資料與案例、5.3 評估指標。\n產出：實驗矩陣、預先宣告分析計畫。"),
    ("公平調參", "1. 只在訓練案例對所有方法做有限且對稱的參數調整。\n2. 鎖定正式參數與預算，保存設定 hash。\n3. 執行小批次壓力測試，估算全部實驗所需機時。", "第 5 章撰寫參數設定與硬體環境。\n產出：參數表、機時估算、正式批次腳本。"),
    ("小型正式實驗", "1. 啟動小型案例正式批次，目標每條件 30 個配對種子。\n2. 每日檢查中止、NaN、覆蓋與連通錯誤，不看結果挑 seed。\n3. 保存原始輸出與不可修改的 manifest。", "第 5 章填入小型案例表格，不先下結論。\n產出：第一批正式原始資料、完整性檢查。"),
    ("中型正式實驗", "1. 執行中型案例正式批次。\n2. 比較每 10000 次評估時間、總執行時間與記憶體。\n3. 將異常分類為程式錯誤、不可行案例或計算資源問題。", "第 5 章更新效率與穩定性表格。\n產出：第二批正式資料、異常清單。"),
    ("大型正式實驗", "1. 執行大型案例或高密度案例。\n2. 確認 RL 推論成本不抵銷搜尋收益。\n3. 若機時不足，依預先規則減少次要案例，不減少主要案例 seed。", "第 5 章撰寫規模擴張結果架構。\n產出：第三批正式資料、資源使用報告。"),
    ("密度與拓樸", "1. 測試不同 sensor target 比例與集中、均勻拓樸。\n2. 分析失敗是覆蓋不足、連通失敗、路由無效或能量耗盡。\n3. 檢查 RL 是否只適用單一拓樸。", "第 5 章完成情境敏感度段落初稿。\n產出：密度與拓樸結果圖、失敗類型圖。"),
    ("能量與移動", "1. 測試初始能量均勻與異質情境。\n2. 以既有 moving 流程比較不移動與允許移動，不把少量移動事件當 RL 主訓練來源。\n3. 記錄每次重新最佳化與移動後的壽命貢獻。", "第 3 章補齊移動機制；第 5 章新增能量與移動結果。\n產出：能量異質性及移動敏感度資料。"),
    ("感測模式", "1. 比較離散感測等級與連續感測斷點。\n2. 確認兩種模式使用相同物理能量公式與公平動作預算。\n3. 分析連續模式改善來自較細半徑還是搜尋空間改變。", "第 3 章完成 sensing mode 定義；第 5 章完成感測模式比較。\n產出：離散連續消融與半徑分布圖。"),
    ("Fitness 消融", "1. 比較原始 fitness 與距離感知或負載平衡版本，前提是此項仍在研究範圍。\n2. 固定搜尋演算法，隔離 fitness 改變造成的影響。\n3. 檢查較高 fitness 是否真的對應較長 lifetime。", "第 4 章或附錄說明 fitness 版本；第 5 章撰寫消融。\n產出：fitness lifetime 相關圖與版本選擇依據。"),
    ("State 消融", "1. 逐組移除停滯、能量、可行性、多樣性與搜尋進度特徵。\n2. 比較完整 state、最小 state 與只用時間進度。\n3. 分析各特徵對 action 選擇與 lifetime 的貢獻。", "第 4 章補上 state 選擇理由；第 5 章新增 state 消融。\n產出：state 消融表、特徵敏感度圖。"),
    ("Action Reward 消融", "1. 比較 RL 控制器、固定算子比例、隨機選擇與手工 phase schedule。\n2. 比較有無評估成本懲罰及不同 reward 視窗。\n3. 檢查提升是否來自更多計算量。", "第 5 章完成 action 與 reward 消融初稿。\n產出：控制策略比較與 action 軌跡圖。"),
    ("泛化實驗", "1. 在未參與訓練的新規模、拓樸或種子評估凍結控制器。\n2. 不在測試案例繼續調參。\n3. 比較重新訓練、直接移轉與固定策略。", "第 5 章撰寫泛化結果與限制。\n產出：train test 分割表、泛化結果。"),
    ("統計分析", "1. 先畫配對差值、分布與信賴區間。\n2. 兩方法用配對檢定與 effect size；多方法用 Friedman 或適合的重複量測方法及事後比較。\n3. 同時報告 lifetime、時間、評估次數與成功率。", "第 5 章填入統計結果與圖表。\n產出：可重跑統計程式、圖表、主結果表。\n里程碑：2027/4/16 主實驗結果凍結。"),
    ("異常重跑", "1. 僅重跑有明確技術理由的遺失或損壞實驗，保存原紀錄。\n2. 不刪除不利結果。\n3. 檢查結論對極端 seed 與不同統計方式是否穩健。", "第 5 章完成主要結果 0.7 版。\n產出：重跑稽核表、穩健性分析。"),
    ("失敗案例", "1. 選代表性成功與失敗軌跡。\n2. 分析 RL 在何種 state 選錯 action，以及失敗對覆蓋、連通與能量的影響。\n3. 形成可被論文討論的限制，而非只列平均值。", "第 5 章完成討論與失敗案例；第 6 章建立限制清單。\n產出：個案圖、限制與適用條件。"),
    ("結果整合", "1. 統一圖表字體、座標、單位、顏色與誤差表示。\n2. 完成時間複雜度與實測 overhead 比較。\n3. 每個研究問題明確對應一張主表或主圖。", "第 5 章完成 0.9 版；第 6 章完成結論骨架。\n產出：最終圖表清單、RQ 證據對照表。"),
    ("全文初稿", "1. 合併六章，檢查符號、名詞、演算法版本與數字一致。\n2. 完成摘要、關鍵詞、圖表目錄與參考文獻。\n3. 每個主張回連到結果或文獻。", "六章全文 1.0 版。\n產出：可供指導教授完整閱讀的 Word 或 LaTeX 稿。\n里程碑：2027/5/14 全文初稿。"),
    ("第一輪修改", "1. 依指導教授意見分為必改、需補證據、文字修正。\n2. 優先修正研究問題、方法描述與公平性。\n3. 只有必要時補小型實驗，不開新支線。", "修訂第 1 至第 4 章，完成回覆意見表。\n產出：全文 1.1 版、修改追蹤表。"),
    ("第二輪修改", "1. 修訂結果解讀，避免把相關誤寫成因果。\n2. 確認結論不超出實驗範圍。\n3. 完成中英文摘要與貢獻條列。", "修訂第 5 至第 6 章。\n產出：全文 1.2 版、摘要定稿候選。"),
    ("格式與送審", "1. 套用學校論文格式，檢查頁碼、目錄、圖表標題、引用與附錄。\n2. 準備口試委員名單、可行時段、申請文件與論文寄送版本。\n3. 執行相似度與引用完整性自查。", "全文 2.0 送審版。\n產出：格式檢核表、口試申請文件草稿、委員版論文。"),
    ("口試申請", "1. 最晚 6/8 完成所有簽核資料，預留行政退件修正時間。\n2. 6/10 前完成口試申請或提出程序，並將論文送交委員。\n3. 方法與主結果正式凍結，只修正致命錯誤。", "完成委員版論文與簡報故事線。\n產出：口試申請完成證明、送件紀錄。\n硬期限：2027/6/10。"),
    ("簡報與問答", "1. 製作 20 至 25 分鐘簡報，主線為問題、缺口、方法、證據、限制。\n2. 準備 30 題問答，涵蓋 RL 必要性、公平比較、state reward、統計與複雜度。\n3. 完成第一次計時演練與錄影檢討。", "將每章濃縮為 1 至 3 張核心投影片。\n產出：簡報 0.8 版、問答表、超時刪減清單。"),
    ("模擬口試", "1. 與實驗室或同學完成至少一次正式模擬口試。\n2. 逐頁修正看不清、講不完、缺乏證據的內容。\n3. 與指導教授確認正式口試日期，目標不晚於 6/25。", "完成論文小錯與簡報修訂。\n產出：簡報 1.0 版、模擬口試意見回覆表。"),
    ("正式口試", "1. 優先於本週完成正式口試；若日期較晚，至少完成第二次全程演練。\n2. 口試當日記錄委員每項修正與是否需再確認。\n3. 當日備份簡報、論文、原始結果與展示程式。", "產出：口試完成紀錄、委員修正清單、簽名文件。\n期限建議：2027/7/2 前完成正式口試。"),
    ("論文定稿", "1. 逐項回覆委員意見，重大修改先請指導教授確認。\n2. 完成論文上傳前格式、授權、浮水印與圖表解析度檢查。\n3. 取得必要簽名並進行圖書館上傳預檢。", "六章定稿與最終中英文摘要。\n產出：論文最終版、委員意見回覆表、上傳檔。"),
    ("離校作業", "1. 7/9 前完成論文上傳、紙本或系所要求、借物與帳務確認。\n2. 封存程式、環境、設定、原始資料、統計與圖表產生流程。\n3. 7/10 完成離校檢核，不把任何核心行政事項留到當天才開始。", "產出：離校完成證明、研究封存包、交接 README。\n硬期限：2027/7/10。"),
]


def week_dates():
    start = date(2026, 9, 12)
    end = date(2027, 7, 10)
    current = start
    ranges = []
    while current <= end:
        week_end = min(current + timedelta(days=6), end)
        if current.year == week_end.year:
            label = f"{current.year}/{current.month}/{current.day}\n至 {week_end.month}/{week_end.day}"
        else:
            label = f"{current.year}/{current.month}/{current.day}\n至 {week_end.year}/{week_end.month}/{week_end.day}"
        ranges.append(label)
        current = week_end + timedelta(days=1)
    return ranges


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT_NAME
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.line_spacing = 1.25

    for style_name, size in (("Title", 25), ("Heading 1", 17), ("Heading 2", 13)):
        style = styles[style_name]
        style.font.name = FONT_NAME
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLACK)

    for section in doc.sections:
        section.footer_distance = Inches(0.25)
        add_page_number(section.footer.paragraphs[0])


def build_document():
    if len(WEEK_CONTENT) != 44:
        raise ValueError(f"Weekly plan must contain 44 entries, got {len(WEEK_CONTENT)}")

    doc = Document()
    configure_document(doc)

    cover_spacer = doc.add_paragraph()
    cover_spacer.paragraph_format.space_after = Pt(70)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(18)
    remove_paragraph_border(title)
    run = title.add_run("RL 導引超啟發式 WSN 研究與論文完成計畫")
    set_run_font(run, 25, bold=True, color=BLACK)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(28)
    run = subtitle.add_run("從題目選定、實驗設計、六章寫作到口試與離校")
    set_run_font(run, 15, color=BLACK)

    for text, size, bold in (
        ("規劃期間  2026 年 9 月 12 日至 2027 年 7 月 10 日", 12, True),
        ("口試申請目標  2027 年 6 月 10 日前完成", 11, False),
        ("離校目標  2027 年 7 月 10 日完成", 11, False),
        ("暫定題目  強化學習導引超啟發式搜尋之 WSN 或 MWSN 壽命最佳化", 11, False),
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(9)
        r = p.add_run(text)
        set_run_font(r, size, bold=bold)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(50)
    p.paragraph_format.line_spacing = 1.3
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("本計畫將論文寫作與程式、先導實驗、正式實驗同步推進。研究方法在 2027 年 1 月中旬凍結，主實驗在 4 月中旬凍結，5 月中旬交付全文初稿，以保留口試與行政緩衝。")
    set_run_font(r, 11, color=MID_GRAY)

    doc.add_page_break()

    add_heading(doc, "一 計畫結論與執行原則", 1)
    add_body(doc, "建議把論文主貢獻定為 RL 導引超啟發式搜尋策略，由 RL 根據搜尋進度、停滯、可行性、能量與多樣性等狀態，選擇搜尋算子或切換探索與局部搜尋階段。專案中的連續感測半徑、distance aware fitness、encoding 與 decoder 改良可作為問題模型或消融變因，但不宜同時都宣稱為主要創新。")
    add_body(doc, "這個方向比 RL 控制 moving 更適合目前專案。moving 通常只在求解失敗後觸發，單次網路生命週期可提供的互動次數偏少；搜尋迭代則能產生較多且密集的 transition，也較容易用相同計算預算和現有 SA SETS、GI GOMEA 比較。")
    add_body(doc, "每週五或該週最後一個工作日進行驗收。未完成工作不可只向後堆疊，必須決定補做、縮小範圍或移除。正式實驗開始後，任何方法或參數修改都要建立新版本，舊結果不得混入新版本統計。")

    add_heading(doc, "二 關鍵里程碑", 1)
    milestone_rows = [
        ("2026/10/09", "題目與研究範圍凍結", "題目、RQ、主貢獻、比較方法、排除項目均取得指導教授確認"),
        ("2026/11/06", "基準線介面凍結", "資料、預算、seed、metrics、decoder、mutation、fitness 與輸出格式可重現"),
        ("2026/12/04", "RL 最小版本", "能關閉或啟用 RL，並完成 5 seed 的首次配對比較"),
        ("2027/01/15", "方法凍結", "state、action、reward、控制器與正式消融清單固定"),
        ("2027/02/12", "正式實驗凍結", "案例、基準、參數、seed、統計方法與機時確認"),
        ("2027/04/16", "主結果凍結", "主實驗完整，完成 effect size、信賴區間與失敗類型分析"),
        ("2027/05/14", "六章全文初稿", "摘要、六章、圖表、參考文獻與附錄可供完整審閱"),
        ("2027/06/10", "口試申請完成", "完成系所要求之申請、簽核與委員送件"),
        ("2027/07/02", "建議最晚口試日", "保留至少一週修改、上傳與簽核時間"),
        ("2027/07/10", "離校完成", "論文上傳、文件、借物、帳務及研究封存全部完成"),
    ]
    add_table(doc, ["日期", "里程碑", "完成標準"], [1.15, 1.65, 4.7], milestone_rows,
              font_size=9.7, alignments=["center", "center", "left"])

    add_heading(doc, "三 題目選擇與研究界線", 1)
    add_body(doc, "推薦題目暫定為「強化學習導引超啟發式搜尋於具連續感測範圍之移動式無線感測網路壽命最佳化」。若連續感測只屬於模型修正而沒有獨立實驗貢獻，正式題目應縮短為「強化學習導引超啟發式搜尋之移動式無線感測網路壽命最佳化」。")
    topic_rows = [
        ("RL 選擇搜尋算子或階段", "高", "每次搜尋可產生大量 transition，能解釋初期探索與後期局部搜尋的選擇", "主要題目"),
        ("RL 自適應 fitness 權重", "中", "可研究能量平衡與覆蓋，但 reward 與真實 lifetime 之間可能不一致", "次要或未來工作"),
        ("RL 控制 sensing level", "中", "動作與連續半徑直接相關，但動作空間與可行性修復較複雜", "通過先導後再納入"),
        ("RL 控制 moving", "低", "單次生命週期的 moving 次數少，樣本不足，容易學不到穩定策略", "備案或未來工作"),
    ]
    add_table(doc, ["候選方向", "建議度", "判斷理由", "研究角色"], [1.5, 0.65, 3.9, 1.45], topic_rows,
              font_size=9.5, alignments=["left", "center", "left", "center"])

    add_heading(doc, "四 研究問題與成功標準", 1)
    rq_items = [
        "RQ1  RL 導引策略在相同計算預算下，能否提升 WSN 或 MWSN 的 network lifetime。",
        "RQ2  RL 在搜尋初期、中期與後期會選擇哪些算子或搜尋階段，是否符合探索轉向局部搜尋的需求。",
        "RQ3  停滯、能量、可行性、多樣性與搜尋進度等 state 特徵，哪些對決策真正有貢獻。",
        "RQ4  凍結後的 RL 控制器能否泛化到未參與訓練的網路規模、拓樸、seed 與感測模式。",
        "RQ5  RL 的推論與訓練成本是否低於它節省的 fitness evaluations 或執行時間。",
    ]
    for item in rq_items:
        add_bullet(doc, item)
    add_body(doc, "主要成功標準應是多數核心案例的 lifetime 有穩定配對改善，且信賴區間與 effect size 支持實質差異。若 lifetime 相當，但 RL 明顯減少 evaluations 或執行時間，也可形成有效貢獻。只在少數 seed 取得極端高值，不能作為主要結論。", bold_lead="主要成功標準")

    doc.add_page_break()
    add_heading(doc, "五 建議的 RL 設計", 1)
    design_rows = [
        ("State", "搜尋預算進度、連續未改善次數、近期 best 與 mean fitness 斜率、可行解比例、未覆蓋 target 數、斷線數、平均與最差剩餘能量比、active sensor 比例、平均 sensing radius、解多樣性、前一 action"),
        ("Action", "優先選擇現有搜尋算子，或選擇 explore、exploit、repair 三類搜尋階段。先保持小而固定的離散 action space，再評估是否需要更細控制"),
        ("Reward", "使用正規化的 best fitness 或 lifetime proxy 改善，扣除額外 evaluations、執行時間與不可行懲罰；同時保留階段結束回饋，避免只鼓勵短期改善"),
        ("Controller", "先用 tabular Q learning 或 contextual bandit 驗證可學性。只有在 transition 足夠、state 離散化造成明顯資訊損失時，才升級 DQN 或其他深度方法"),
        ("Training", "訓練案例與測試案例分離；多個網路實例共同訓練；checkpoint、epsilon、seed、episode 與 feature normalization 全部寫入輸出"),
    ]
    add_table(doc, ["元件", "建議內容"], [1.15, 6.35], design_rows, font_size=9.8,
              alignments=["center", "left"])

    add_heading(doc, "六 六章論文架構與完成標準", 1)
    chapter_rows = [
        ("第 1 章 緒論", "研究背景、WSN 或 MWSN 壽命問題、現有固定超啟發式策略的限制、研究動機、研究問題、貢獻與章節安排", "8 至 12 頁；每個貢獻能對應方法與實驗證據", "2027/05/14"),
        ("第 2 章 文獻探討", "WSN 覆蓋連通路由與能量、壽命最佳化、超啟發式、GOMEA 與 SA SETS 類方法、RL 導引最佳化、2021 至 2026 年相關研究與研究缺口", "15 至 20 頁；文獻分類清楚，避免只逐篇摘要", "2026/12/25 初稿"),
        ("第 3 章 系統模型與問題定義", "節點與能量模型、sensing radius、coverage、connectivity、routing、scheduler、moving、encoding decoding、目標函數與限制條件", "15 至 20 頁；符號、公式、程式行為一致，離散與連續模式可重現", "2026/12/04 初稿"),
        ("第 4 章 RL 導引超啟發式方法", "整體架構、state、action、reward、controller、訓練與推論流程、與 SA SETS 或其他母演算法的整合、虛擬碼、複雜度", "18 至 25 頁；關閉 RL 能回到公平基準，所有設計選擇有理由", "2027/01/15 方法版"),
        ("第 5 章 實驗與結果分析", "RQ、案例、基準、參數、硬體、指標、統計、主結果、消融、泛化、時間成本、成功與失敗案例、限制", "25 至 35 頁；每個 RQ 至少有一項主證據，原始資料與程式可重跑", "2027/04/30 結果版"),
        ("第 6 章 結論與未來工作", "研究發現、貢獻回應、適用條件、限制、未來可研究的 moving、continuous control、GPU 或多目標方向", "5 至 8 頁；不提出未被實驗支持的新結論", "2027/05/14 初稿"),
    ]
    add_table(doc, ["章別", "內容", "完成標準", "目標日"], [1.25, 2.75, 2.35, 1.05], chapter_rows,
              font_size=9.0, alignments=["left", "left", "left", "center"])

    add_heading(doc, "七 正式實驗設計", 1)
    add_heading(doc, "七一 比較方法", 2)
    add_bullet(doc, "固定策略基準：目前穩定的 SA SETS 主版本，完整記錄 SensorEncoding decoder、mutation 與 fitness service 版本。")
    add_bullet(doc, "專案內強基準：GI GOMEA 及先導實驗中表現最穩定的 SA SETS 變體。")
    add_bullet(doc, "控制策略基準：固定算子比例、手工 phase schedule、隨機 action，以及 RL 導引版本。")
    add_bullet(doc, "其他演算法只在程式穩定、使用相同解碼與停止預算時納入，避免為了數量犧牲公平性。")

    add_heading(doc, "七二 案例與指標", 2)
    experiment_rows = [
        ("網路規模", "小、中、大；依現有資料集或 problem generator 的可重現設定"),
        ("情境變因", "sensor target 密度、拓樸分布、初始能量異質性、離散或連續感測、是否允許 moving"),
        ("主要指標", "network lifetime；若 lifetime 定義有多種，正式實驗前固定唯一主定義"),
        ("次要指標", "coverage、connectivity、總 cost、最差區域耗能比、active sensors、平均 radius、成功排程長度"),
        ("效率指標", "fitness evaluations、每 10000 次評估秒數、總時間、RL overhead、記憶體"),
        ("重複次數", "先導 5 至 10 seeds；核心正式案例建議 30 個配對 seeds；次要敏感度依機時可採 15 至 20"),
        ("統計", "報告中位數與平均數、95 percent 信賴區間、配對差值、effect size；多方法比較需控制多重比較"),
    ]
    add_table(doc, ["項目", "規格"], [1.45, 6.05], experiment_rows, font_size=9.6,
              alignments=["center", "left"])

    add_heading(doc, "七三 每次實驗必存欄位", 2)
    add_body(doc, "每次 run 至少保存 experiment id、algorithm、RL controller、seed、problem instance、sensing mode、SensorEncoding decoder、mutation version、fitness service、權重、搜尋預算、停止條件、程式 commit 或版本識別、Python 與套件環境、CPU 或 GPU、開始結束時間、lifetime、fitness、cost、evaluations、reoptimizations、失敗原因與原始 trace。這可避免日後無法判斷某批結果使用 mutatev1 或 mutatev2。")

    add_heading(doc, "八 逐週研究與論文時程", 1)
    add_body(doc, "下表第一欄是第幾週，第二欄是日期範圍。每週工作均附論文章節與可驗收產出；若學校正式期限不同，優先調整第 39 至第 44 週，並保留至少七天的口試後修改時間。")
    date_ranges = week_dates()
    weekly_rows = []
    for index, ((phase, work, output), date_label) in enumerate(zip(WEEK_CONTENT, date_ranges), start=1):
        weekly_rows.append((f"第 {index} 週", date_label, phase, work, output))
    chunk_sizes = [4, 6, 6, 6, 6, 6, 6, 4]
    start = 0
    for chunk_index, chunk_size in enumerate(chunk_sizes):
        if chunk_index > 0:
            start_new_page(doc)
        end = start + chunk_size
        add_table(doc, ["週次", "日期", "階段", "主要工作", "論文與驗收產出"],
                  [0.55, 0.95, 1.05, 3.15, 1.8], weekly_rows[start:end], font_size=8.7,
                  alignments=["center", "center", "center", "left", "left"])
        start = end

    start_new_page(doc)
    add_heading(doc, "九 每週固定節奏", 1)
    rhythm_rows = [
        ("週一", "確認本週唯一主要成果；拆成可在兩至四小時完成的工作單位；啟動或續跑長時間實驗"),
        ("週二至週三", "程式、實驗與分析主工作；每次修改必有版本與測試；寫下當日觀察，不只保存 CSV"),
        ("週四", "更新論文章節、圖表與文獻；將本週程式結果轉為可閱讀的表或段落"),
        ("週五", "驗收產出、備份、檢查失敗工作；準備給指導教授的一頁進度與明確問題"),
        ("週末", "保留半天緩衝或休息；只有長時間批次可持續執行，避免用週末填補長期超載"),
    ]
    add_table(doc, ["時間", "固定工作"], [1.15, 6.35], rhythm_rows, font_size=9.8,
              alignments=["center", "left"])

    add_heading(doc, "十 指導教授會議建議", 1)
    add_body(doc, "前四週建議每週或每兩週確認一次方向；方法整合期至少每兩週一次；正式實驗期每兩週提供固定格式摘要；5 月起每週回報論文與口試事項。每次會議只帶一份一頁摘要，包含本週結論、證據、尚未解決問題、下週承諾與需要教授決定的事項。")
    for q in (
        "這篇論文的唯一主要創新，是否確定為 RL 選擇搜尋算子或階段。",
        "連續感測半徑要放入題目、方法貢獻，還是只作模型與消融。",
        "正式 comparison 必須包含哪些演算法，學校或教授是否要求特定 baseline。",
        "6 月 10 日是口試申請截止、論文送委員截止，還是正式口試日。",
        "系所要求的論文格式、相似度門檻、口試委員邀請與離校流程為何。",
    ):
        add_bullet(doc, q)

    add_heading(doc, "十一 風險與備案", 1)
    risk_rows = [
        ("題目過大", "RL、連續感測、distance fitness、GPU 與 moving 同時成為主貢獻", "10 月鎖定一項主貢獻；其他只作消融或未來工作"),
        ("RL 樣本不足", "action 沒有學習趨勢，或每次求解 transition 太少", "由 moving 改為搜尋迭代控制；跨實例訓練；先用 bandit 或 tabular 方法"),
        ("基準漂移", "decoder、mutation 或 fitness 改動後仍沿用舊結果", "每 run 保存完整版本欄位；版本不同即分開分析"),
        ("實驗太慢", "正式矩陣機時超過可用時間", "2 月先估機時；保留核心 30 seed，刪減次要情境，不犧牲主要比較"),
        ("RL 未提升 lifetime", "平均差異小或信賴區間跨零", "檢查是否能以較少 evaluations 達到相同 lifetime；誠實報告適用情境與負結果"),
        ("寫作延遲", "4 月後才開始整理模型與方法", "第 3 章在 12 月、第 4 章在 1 月先完成；每週把實驗結果轉成論文素材"),
        ("行政時間不足", "口試或上傳退件後沒有修正時間", "6/8 前完成申請資料；正式口試目標不晚於 6/25；7/9 前完成離校核心事項"),
    ]
    add_table(doc, ["風險", "警訊", "立即措施"], [1.15, 3.0, 3.35], risk_rows, font_size=9.4,
              alignments=["center", "left", "left"])

    add_heading(doc, "十二 口試與離校完成清單", 1)
    checklist_items = [
        "□ 題目、中文摘要、英文摘要與系統登錄資料一致。",
        "□ 口試委員與時間完成確認，論文依規定提前送達。",
        "□ 六章內容、目錄、圖表目錄、引用、附錄與頁碼通過格式檢查。",
        "□ 所有主結果可由固定指令重現，原始資料與統計程式均已備份。",
        "□ 口試委員修正逐項完成並取得指導教授確認。",
        "□ 電子論文上傳、授權書、簽名頁、紙本與系所文件完成。",
        "□ 借物、帳務、門禁、研究室設備與其他離校程序完成。",
        "□ 程式、environment、資料、設定、論文圖表與 README 完成封存。",
    ]
    for item in checklist_items:
        add_bullet(doc, item)

    add_heading(doc, "十三 第一週立即行動", 1)
    for number, item in enumerate((
        "向系所確認 2027 年口試申請、正式口試、電子論文與離校的真正截止日。",
        "把目前所有 experiment_results 依演算法、decoder、mutation、fitness 與 sensing mode 建立索引。",
        "用一頁寫出目前最佳基準、RL 想控制的對象、預期 state action reward 與三個最主要風險。",
        "約指導教授在 10 月 9 日前完成題目與研究範圍確認。",
        "建立每週進度檔，固定記錄完成項、證據、問題、決定與下一週工作。",
    ), start=1):
        add_numbered(doc, item, number)

    core = doc.core_properties
    core.title = "RL 導引超啟發式 WSN 研究與論文完成計畫"
    core.subject = "2026 年 9 月至 2027 年 7 月研究 實驗 六章論文 口試與離校時程"
    core.author = "WSN 研究計畫"
    core.keywords = "WSN, MWSN, reinforcement learning, metaheuristic, thesis plan"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH)
    return OUTPUT_PATH


def build_table_document():
    """Build the user-facing version containing one clear weekly table only."""
    if len(WEEK_CONTENT) != 44:
        raise ValueError(f"Weekly plan must contain 44 entries, got {len(WEEK_CONTENT)}")

    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(10)
    remove_paragraph_border(title)
    run = title.add_run("RL 導引超啟發式 WSN 論文週次規劃")
    set_run_font(run, 21, bold=True, color=BLACK)
    run.add_break()
    period = title.add_run("2026 年 9 月 12 日至 2027 年 7 月 10 日")
    set_run_font(period, 11, color=MID_GRAY)

    dates = week_dates()
    rows = []
    for index, ((phase, work, output), date_label) in enumerate(zip(WEEK_CONTENT, dates), start=1):
        rows.append((f"第 {index} 週", date_label, phase, work, output))

    # These are one logical schedule table. It is split only at page boundaries
    # so Word renders a complete header on every page without clipped repeats.
    chunk_sizes = [6, 7, 8, 8, 8, 7]
    start = 0
    for chunk_index, chunk_size in enumerate(chunk_sizes):
        if chunk_index > 0:
            start_new_page(doc)
        end = start + chunk_size
        add_table(
            doc,
            ["週次", "日期", "階段", "研究與實驗工作", "論文寫作與驗收"],
            [0.55, 0.95, 1.05, 3.15, 1.8],
            rows[start:end],
            font_size=8.7,
            alignments=["center", "center", "center", "left", "left"],
        )
        start = end

    core = doc.core_properties
    core.title = "RL 導引超啟發式 WSN 論文週次規劃"
    core.subject = "2026 年 9 月 12 日至 2027 年 7 月 10 日研究 實驗 六章論文 口試與離校規劃"
    core.author = "WSN 研究計畫"
    core.keywords = "WSN, MWSN, reinforcement learning, metaheuristic, thesis plan"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build_table_document())

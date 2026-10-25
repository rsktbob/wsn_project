from datetime import date, timedelta
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


SOURCE = Path(r"C:\Users\bob10\Desktop\WSN研究與論文時程規劃.docx")
OUTPUT = Path(r"C:\wsn研究\wsn_project\artifacts\WSN研究與論文時程規劃_日期週次修正.docx")
START = date(2026, 9, 12)
END = date(2027, 7, 10)


def date_range(first_week: int, last_week: int) -> str:
    start = START + timedelta(days=(first_week - 1) * 7)
    end = min(START + timedelta(days=last_week * 7 - 1), END)
    if start == end:
        return f"{start.year}/{start.month}/{start.day}"
    start_text = f"{start.year}/{start.month}/{start.day}"
    if len(start_text) > 9:
        start_text = f"{start.year}/{start.month}/\n{start.day}"
    finish = (
        f"{end.month}/{end.day}"
        if start.year == end.year
        else f"{end.year}/{end.month}/{end.day}"
    )
    return f"{start_text}\n至 {finish}"


def set_cell_text(cell, value: str, font_size: int | None = None) -> None:
    paragraph = cell.paragraphs[0]
    for run in paragraph.runs:
        run.text = ""
    run = paragraph.add_run(value)
    if font_size is not None:
        run.font.size = Pt(font_size)
    for extra in cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)


def label(first_week: int, last_week: int | None = None) -> str:
    if last_week is None or first_week == last_week:
        return f"第 {first_week} 週"
    return f"第 {first_week} 至 {last_week} 週"


def update_row(row, first_week: int, last_week: int | None = None) -> None:
    last_week = first_week if last_week is None else last_week
    set_cell_text(row.cells[0], label(first_week, last_week))
    set_cell_text(row.cells[1], date_range(first_week, last_week), font_size=10)


document = Document(SOURCE)
first_table, second_table = document.tables

for row, first_week, last_week in zip(
    first_table.rows[1:],
    [1, 3, 4, 8, 9, 10, 12, 13, 16],
    [2, 3, 7, 8, 9, 11, 12, 15, 17],
):
    update_row(row, first_week, last_week)

for row, first_week, last_week in zip(
    second_table.rows[1:11],
    [18, 22, 26, 30, 31, 33, 34, 39, 40, 44],
    [21, 25, 29, 30, 32, 33, 39, 39, 43, 44],
):
    update_row(row, first_week, last_week)

# 第 39 週的申請期限是當週的里程碑；保留明確日期，避免誤認為下一週。
set_cell_text(second_table.rows[8].cells[0], "第 39 週\n申請口試")
set_cell_text(second_table.rows[8].cells[1], "2027/6/10", font_size=10)

# 原表最後一列排到 7/11 之後；時程以 7/10 離校為終點，因此移除。
second_table._tbl.remove(second_table.rows[11]._tr)
set_cell_text(second_table.rows[10].cells[2], "口試與離校")

# 原檔在兩張表格之間強制換頁，會產生只有一列的空白頁；讓表格自然續排。
for paragraph in document.paragraphs:
    page_break = paragraph._p.pPr.find(qn("w:pageBreakBefore")) if paragraph._p.pPr is not None else None
    if page_break is not None:
        paragraph._p.pPr.remove(page_break)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
document.save(OUTPUT)
print(OUTPUT)

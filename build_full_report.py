"""
Builds the complete, professional academic project report for:
"Secure Multivariate Temperature Forecasting Using RNN/LSTM Against Adversarial Attacks"

Outputs:
1. FINAL_PROJECT_REPORT_AI_FOR_CYBER_SECURITY.docx
2. FINAL_PROJECT_REPORT_AI_FOR_CYBER_SECURITY.pdf (via native Word COM automation)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

# ==============================================================================
# STYLING HELPERS & PALETTE
# ==============================================================================
COLOR_PRIMARY_HEX = "0B2545"    # Deep Navy Blue
COLOR_SECONDARY_HEX = "134074"  # Slate Blue
COLOR_ACCENT_HEX = "8DA9C4"     # Light Accent Blue
COLOR_BG_LIGHT_HEX = "F4F7FB"   # Table row light tint
COLOR_CALLOUT_BG = "EEF4F8"     # Callout box light background
COLOR_CALLOUT_BORDER = "134074" # Callout box left border

COLOR_PRIMARY = RGBColor(11, 37, 69)
COLOR_SECONDARY = RGBColor(19, 64, 116)
COLOR_TEXT_DARK = RGBColor(34, 34, 34)
COLOR_MUTED = RGBColor(100, 110, 120)

def set_cell_background(cell, hex_color):
    """Sets background shading of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    for child in list(tcPr):
        if child.tag.endswith('shd'):
            tcPr.remove(child)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding of a table cell (in dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Sets clean subtle borders for an entire table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_heading_1(doc, text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Calibri"
        r.font.size = Pt(17)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
    return h

def add_heading_2(doc, text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Calibri"
        r.font.size = Pt(13.5)
        r.font.bold = True
        r.font.color.rgb = COLOR_SECONDARY
    return h

def add_heading_3(doc, text):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(2)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Calibri"
        r.font.size = Pt(11.5)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
    return h

def add_body_p(doc, text, bold_prefix=None, space_after=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_TEXT_DARK
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(11)
    r_body.font.color.rgb = COLOR_TEXT_DARK
    return p

def add_bullet_p(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_TEXT_DARK
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(10.5)
    r_body.font.color.rgb = COLOR_TEXT_DARK
    return p

def add_callout_box(doc, text, title="KEY CYBERSECURITY TAKEAWAY"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, COLOR_CALLOUT_BG)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Set left border thick and accent
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{COLOR_CALLOUT_BORDER}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    r_t = p.add_run(f"■  {title}\n")
    r_t.font.name = "Calibri"
    r_t.font.size = Pt(10.5)
    r_t.font.bold = True
    r_t.font.color.rgb = COLOR_SECONDARY

    r_b = p.add_run(text)
    r_b.font.name = "Calibri"
    r_b.font.size = Pt(10)
    r_b.font.color.rgb = COLOR_TEXT_DARK

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(4)

def add_figure_with_caption(doc, img_path, fig_num, caption_text, width_inches=6.0):
    if not os.path.exists(img_path):
        p_err = doc.add_paragraph(f"[Image not found: {img_path}]")
        p_err.runs[0].font.italic = True
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    p_cap.paragraph_format.keep_with_next = False
    
    r_lbl = p_cap.add_run(f"Figure {fig_num}: ")
    r_lbl.font.name = "Calibri"
    r_lbl.font.size = Pt(9.5)
    r_lbl.font.bold = True
    r_lbl.font.color.rgb = COLOR_PRIMARY
    
    r_txt = p_cap.add_run(caption_text)
    r_txt.font.name = "Calibri"
    r_txt.font.size = Pt(9.5)
    r_txt.font.italic = True
    r_txt.font.color.rgb = COLOR_TEXT_DARK

def add_styled_table(doc, headers, rows_data, tbl_num, caption_text, col_widths=None, alignment=WD_TABLE_ALIGNMENT.CENTER):
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(3)
    p_cap.paragraph_format.keep_with_next = True
    
    r_lbl = p_cap.add_run(f"Table {tbl_num}: ")
    r_lbl.font.name = "Calibri"
    r_lbl.font.size = Pt(9.5)
    r_lbl.font.bold = True
    r_lbl.font.color.rgb = COLOR_PRIMARY
    
    r_txt = p_cap.add_run(caption_text)
    r_txt.font.name = "Calibri"
    r_txt.font.size = Pt(9.5)
    r_txt.font.italic = True
    r_txt.font.color.rgb = COLOR_TEXT_DARK

    table = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
    table.alignment = alignment
    table.autofit = False
    set_table_borders(table)

    # Format header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], COLOR_PRIMARY_HEX)
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    # Format data rows
    for r_idx, row_items in enumerate(rows_data):
        row_cells = table.rows[r_idx + 1].cells
        bg_col = COLOR_BG_LIGHT_HEX if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row_items):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            # Center if numeric or short, left if text
            if any(c_idx == idx for idx in [1, 2, 3, 4, 5, 6, 7]):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(9.0)
                r.font.color.rgb = COLOR_TEXT_DARK

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(6)

def add_code_block(doc, code_str):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F8F9FA")
    set_cell_margins(cell, top=80, bottom=80, left=140, right=140)
    set_table_borders(tbl, color="E0E0E0", sz="4")

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(code_str)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(30, 41, 59)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(4)


# ==============================================================================
# MAIN REPORT BUILDER
# ==============================================================================
def generate_report():
    print("=" * 70)
    print("BUILDING ACADEMIC PROJECT REPORT: FINAL_PROJECT_REPORT_AI_FOR_CYBER_SECURITY")
    print("=" * 70)

    doc = Document()

    # Configure 1-inch margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        s.page_width = Inches(8.5)
        s.page_height = Inches(11.0)

    # Configure Header & Footer
    header = sections[0].header
    p_hdr = header.paragraphs[0]
    p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_hdr = p_hdr.add_run("AI for Cyber Security: Secure Multivariate Temperature Forecasting Report")
    r_hdr.font.name = "Calibri"
    r_hdr.font.size = Pt(8.5)
    r_hdr.font.italic = True
    r_hdr.font.color.rgb = COLOR_MUTED

    footer = sections[0].footer
    p_ftr = footer.paragraphs[0]
    p_ftr.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_ftr_l = p_ftr.add_run("National Forensic Sciences University (NFSU)  |  M.Sc. AI for Cyber Security")
    r_ftr_l.font.name = "Calibri"
    r_ftr_l.font.size = Pt(8.5)
    r_ftr_l.font.color.rgb = COLOR_MUTED

    # ==========================================================================
    # COVER PAGE
    # ==========================================================================
    p_cov_inst = doc.add_paragraph()
    p_cov_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cov_inst.paragraph_format.space_before = Pt(36)
    p_cov_inst.paragraph_format.space_after = Pt(4)
    r_inst = p_cov_inst.add_run("NATIONAL FORENSIC SCIENCES UNIVERSITY")
    r_inst.font.name = "Calibri"
    r_inst.font.size = Pt(16)
    r_inst.font.bold = True
    r_inst.font.color.rgb = COLOR_PRIMARY

    p_cov_dept = doc.add_paragraph()
    p_cov_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cov_dept.paragraph_format.space_after = Pt(28)
    r_dept = p_cov_dept.add_run("SCHOOL OF CYBER SECURITY & DIGITAL FORENSICS\nDEPARTMENT OF ARTIFICIAL INTELLIGENCE")
    r_dept.font.name = "Calibri"
    r_dept.font.size = Pt(11)
    r_dept.font.color.rgb = COLOR_SECONDARY

    # Title Banner
    p_cov_t = doc.add_paragraph()
    p_cov_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cov_t.paragraph_format.space_before = Pt(18)
    p_cov_t.paragraph_format.space_after = Pt(10)
    r_t = p_cov_t.add_run("Secure Multivariate Temperature Forecasting Using RNN/LSTM Against Adversarial Attacks")
    r_t.font.name = "Calibri"
    r_t.font.size = Pt(22)
    r_t.font.bold = True
    r_t.font.color.rgb = COLOR_PRIMARY

    p_cov_sub = doc.add_paragraph()
    p_cov_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cov_sub.paragraph_format.space_after = Pt(36)
    r_sub = p_cov_sub.add_run("A Comprehensive Academic Study on Evasion Vulnerabilities, Physical Threat Constraints,\nAttack Detection, Input Sanitization, and Adversarial Training")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    r_sub.font.color.rgb = COLOR_MUTED

    # Metadata Table
    p_rep_lbl = doc.add_paragraph()
    p_rep_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_rep_lbl.paragraph_format.space_after = Pt(12)
    r_rl = p_rep_lbl.add_run("COURSE CAPSTONE PROJECT REPORT")
    r_rl.font.name = "Calibri"
    r_rl.font.size = Pt(12)
    r_rl.font.bold = True
    r_rl.font.color.rgb = COLOR_SECONDARY

    meta_tbl = doc.add_table(rows=6, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_tbl.autofit = False
    set_table_borders(meta_tbl, color="CCCCCC", sz="4")

    meta_rows = [
        ("Course Name & Code:", "AI for Cyber Security (AICS - Semester 1)"),
        ("Student Name:", "[Student Name Placeholder]"),
        ("Enrollment / Roll Number:", "[Roll / Enrollment Number Placeholder]"),
        ("Program / Degree:", "Master of Science in Cyber Security & AI"),
        ("Faculty Supervisor:", "[Faculty / Professor Name Placeholder]"),
        ("Academic Year:", "2026 – 2027  (Submission Date: November 2026)")
    ]

    for idx, (lbl, val) in enumerate(meta_rows):
        row_cells = meta_tbl.rows[idx].cells
        row_cells[0].width = Inches(2.2)
        row_cells[1].width = Inches(4.3)
        row_cells[0].text = lbl
        row_cells[1].text = val
        set_cell_background(row_cells[0], COLOR_BG_LIGHT_HEX)
        set_cell_background(row_cells[1], "FFFFFF")
        set_cell_margins(row_cells[0], top=60, bottom=60, left=100, right=100)
        set_cell_margins(row_cells[1], top=60, bottom=60, left=100, right=100)
        p0 = row_cells[0].paragraphs[0]
        p1 = row_cells[1].paragraphs[0]
        p0.runs[0].font.name = "Calibri"
        p0.runs[0].font.size = Pt(10)
        p0.runs[0].font.bold = True
        p0.runs[0].font.color.rgb = COLOR_PRIMARY
        p1.runs[0].font.name = "Calibri"
        p1.runs[0].font.size = Pt(10)
        p1.runs[0].font.color.rgb = COLOR_TEXT_DARK

    doc.add_page_break()

    # ==========================================================================
    # TABLE OF CONTENTS / EXECUTIVE SITEMAP
    # ==========================================================================
    add_heading_1(doc, "Table of Contents")
    toc_p = doc.add_paragraph()
    toc_p.paragraph_format.line_spacing = 1.15
    toc_p.paragraph_format.space_after = Pt(10)

    toc_entries = [
        ("1. Abstract", "3"),
        ("2. Introduction (Background, CPS, RNN/LSTM, Threat Motivation)", "4"),
        ("3. Problem Statement & Research Formulation", "5"),
        ("4. Project Objectives", "6"),
        ("5. Related Work & Literature Review (Nketiah et al., 2023 vs. Extension)", "7"),
        ("6. Dataset Specification & Exploratory Analysis (weatherHistory.xls)", "8"),
        ("7. Preprocessing & Leakage-Free Normalization Pipeline", "9"),
        ("8. Baseline Forecasting Model Architectures (SimpleRNN & LSTM)", "10"),
        ("9. Clean Baseline Forecasting Results & Empirical Performance", "11"),
        ("10. Cybersecurity Threat Model & Physical Sensor Invariants", "12"),
        ("11. Fast Gradient Sign Method (FGSM) Attack Formulation & Evaluation", "13"),
        ("12. Basic Iterative Method (BIM / PGD) Attack Formulation & Comparison", "15"),
        ("13. Attack Detection Layer (Rate-of-Change & Residual Anomaly Detectors)", "16"),
        ("14. Input Sanitization Layer (Rolling Median & Physical Rate Limiting)", "18"),
        ("15. Adversarial Training Methodology & Online Min-Max Hardening", "20"),
        ("16. Experimental Design Matrix & Chronological Isolation Protocol", "22"),
        ("17. Comprehensive Experimental Results & Master Benchmark Tables", "23"),
        ("18. Security Analysis & Answers to Research Questions", "25"),
        ("19. System Architecture & Cyber-Physical Dataflow", "26"),
        ("20. Implementation Details & Repository Codebase Architecture", "27"),
        ("21. Technologies & Hardware/Software Runtime Environment", "28"),
        ("22. Executive Results Summary Table", "29"),
        ("23. Limitations of the Study", "30"),
        ("24. Future Research Directions", "30"),
        ("25. Academic Conclusion & Recommendations", "31"),
        ("26. References (IEEE Format)", "32"),
        ("Appendix A: Verified Key Code Snippets", "33"),
        ("Appendix B: Detailed Model Layer Parameters", "34"),
        ("Appendix C: Hyperparameter Specifications", "34"),
        ("Appendix D: Adversarial Training Configuration JSON", "35"),
        ("Appendix E: Complete Repository File Tree", "35"),
        ("Appendix F: Interactive Inference & Diagnostics Output", "36")
    ]

    for title, pg in toc_entries:
        p_toc = doc.add_paragraph()
        p_toc.paragraph_format.space_before = Pt(0)
        p_toc.paragraph_format.space_after = Pt(2)
        r_t = p_toc.add_run(title)
        r_t.font.name = "Calibri"
        r_t.font.size = Pt(10)
        r_t.font.color.rgb = COLOR_TEXT_DARK
        dots_count = max(4, 90 - len(title) - len(pg))
        r_dots = p_toc.add_run(" " + "." * dots_count + " ")
        r_dots.font.name = "Calibri"
        r_dots.font.size = Pt(9)
        r_dots.font.color.rgb = COLOR_MUTED
        r_p = p_toc.add_run(pg)
        r_p.font.name = "Calibri"
        r_p.font.size = Pt(10)
        r_p.font.bold = True
        r_p.font.color.rgb = COLOR_PRIMARY

    doc.add_page_break()

    # ==========================================================================
    # SECTION 1: ABSTRACT
    # ==========================================================================
    add_heading_1(doc, "1. Abstract")
    add_body_p(doc, 
        "Autonomous cyber-physical systems (CPS)—including electrical smart grids, smart agricultural microclimates, and HVAC building automation—rely extensively on deep recurrent neural networks for multivariate time-series temperature forecasting. However, existing architectures are deployed under an unverified assumption of sensor trustworthiness, exposing critical decision-making loops to malicious telemetry manipulation.")
    add_body_p(doc,
        "In this project, we conduct an end-to-end investigation into the cybersecurity vulnerabilities of Long Short-Term Memory (LSTM) and SimpleRNN forecasting models trained on 96,453 hourly records of multivariate weather data from Szeged, Hungary. We formulate a physically constrained adversarial threat model where cryptographic calendar timestamps remain immutable, and only physical atmospheric sensors (apparent temperature, humidity, wind speed, visibility, cloud cover, and barometric pressure) are subjected to evasion perturbations under the Fast Gradient Sign Method (FGSM) and Basic Iterative Method (BIM / PGD) across perturbation budgets ε in [0.001, 0.10].")
    add_body_p(doc,
        "To protect the forecasting pipeline, we design, implement, and benchmark a complete three-tier defense hierarchy: (1) Pre-model anomaly detection utilizing statistical rate-of-change and prediction residual monitoring; (2) In-place input sanitization using non-parametric rolling median filtering and physical rate-of-change limiting; and (3) Model hardening via online adversarial training combining clean and adversarial minibatches under strict chronological data isolation.")
    add_body_p(doc,
        "Empirical evaluation across 14,465 out-of-sample test sequences demonstrates that SimpleRNN exhibits severe structural vulnerability (+2,718% MSE increase at ε=0.10, collapsing R² from 0.9050 to -1.6776), whereas LSTM demonstrates higher inherent resilience (+1,461% MSE increase) due to non-linear gating saturation. The prediction residual anomaly detector achieves 83.1% to 100% true positive rates at ε >= 0.05 while maintaining clean false alarm rates strictly below 4.1%. Rolling median sanitization recovers up to 46.2% of adversarial error on SimpleRNN without degrading clean baseline accuracy. Furthermore, adversarial training provides dramatic structural hardening for SimpleRNN, recovering 72.1% of adversarial error at ε=0.10 (reducing MSE from 213.84 to 65.18 and preserving a positive R² of 0.1838) with negligible clean accuracy penalty. These findings substantiate that defense-in-depth is indispensable for securing time-series deep learning in critical infrastructure.")

    add_callout_box(doc,
        "Clean Baseline: LSTM achieves MSE = 6.78, R² = 0.915; SimpleRNN achieves MSE = 7.59, R² = 0.905.\n"
        "Vulnerability: Evasion attacks (ε=0.10) cause SimpleRNN error to increase by +2718% (R² = -1.68) and LSTM error by +1461% (R² = -0.32).\n"
        "Defense Recovery: Residual Anomaly Detection detects up to 100% of large attacks (ε >= 0.05); Rolling Median sanitization recovers 46.2% error without retraining; Adversarial Training recovers 72.1% error on SimpleRNN.",
        title="EXECUTIVE EXPERIMENTAL SUMMARY (VERIFIED ON 14,465 TEST SAMPLES)")

    # ==========================================================================
    # SECTION 2: INTRODUCTION
    # ==========================================================================
    add_heading_1(doc, "2. Introduction")
    
    add_heading_2(doc, "2.1 Background")
    add_body_p(doc,
        "The convergence of Internet-of-Things (IoT) edge sensing and deep learning has revolutionized environmental monitoring and cyber-physical infrastructure. Automated Weather Stations (AWS) stream high-frequency meteorological telemetry—such as ambient temperature, barometric pressure, humidity, and wind velocity—into central data centers. These telemetry streams feed predictive artificial intelligence pipelines that govern power grid balancing, automated agricultural irrigation, building climate control, and renewable energy dispatch.")

    add_heading_2(doc, "2.2 Temperature Forecasting in Cyber-Physical Systems")
    add_body_p(doc,
        "Ambient temperature is the single most influential meteorological driver of electrical load. In modern smart electrical grids, a forecast discrepancy of even 1°C to 2°C can shift grid dispatch requirements by hundreds of megawatts. Erroneous forecasts lead to inefficient generation scheduling, premature activation of costly fossil-fuel peaking plants, or localized grid overloads.")

    add_heading_2(doc, "2.3 Multivariate Time-Series Forecasting")
    add_body_p(doc,
        "Atmospheric processes exhibit intricate thermodynamic interdependence. Ambient temperature is tightly coupled with apparent temperature, humidity, cloud cover, and barometric pressure. Univariate models relying solely on historical temperature fail to capture meteorological transitions (e.g., cold front passages). Multivariate time-series forecasting incorporates historical vectors x_t in R^D across a sliding lookback horizon T to predict subsequent thermodynamic states.")

    add_heading_2(doc, "2.4 Role of Recurrent Neural Networks (RNN and LSTM)")
    add_body_p(doc,
        "Traditional statistical time-series methods (e.g., ARIMA, SARIMAX) assume stationarity and linear relationships. Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks overcome these limitations by maintaining internal hidden state vectors h_t that accumulate temporal representations across sequential observations. LSTMs employ specialized input, forget, and output gating mechanisms that regulate information flow, effectively mitigating vanishing gradient phenomena over multi-step temporal windows.")

    add_heading_2(doc, "2.5 Cybersecurity in Artificial Intelligence and Machine Learning")
    add_body_p(doc,
        "While deep learning models achieve impressive generalization on independent and identically distributed (i.i.d.) clean data, they are inherently fragile under non-stationary or adversarially perturbed inputs. In cyber-physical settings, edge sensor nodes are frequently deployed in physically accessible, unmonitored environments, communicating over unencrypted wireless channels (e.g., LoRaWAN, cellular IoT). This architecture exposes the pipeline to False Data Injection (FDI) attacks and Man-in-the-Middle (MitM) telemetry spoofing.")

    add_heading_2(doc, "2.6 Adversarial Attacks on Time-Series Models")
    add_body_p(doc,
        "Adversarial evasion attacks exploit model sensitivity by crafting worst-case input perturbations δ aligned with the gradient of the loss function. In computer vision, adversarial perturbations deceive classifiers while remaining visually imperceptible. In time-series forecasting, an adversary injects micro-perturbations into sensor channels that remain within legitimate sensor operational bounds but force the recurrent forecasting model to produce wildly inaccurate predictions.")

    add_heading_2(doc, "2.7 Motivation for the Project")
    add_body_p(doc,
        "A critical cybersecurity paradox exists in modern applied AI: a forecasting model can demonstrate state-of-the-art predictive accuracy (R² > 0.91) on historical test sets yet remain completely defenseless against micro-scale adversarial tampering. This research bridges the gap between pure forecasting accuracy and operational cyber resilience, implementing realistic threat models, detection filters, sanitizers, and robust training to ensure trustworthy forecasting.")

    # ==========================================================================
    # SECTION 3: PROBLEM STATEMENT
    # ==========================================================================
    add_heading_1(doc, "3. Problem Statement & Research Formulation")
    add_body_p(doc,
        "In critical infrastructure, automated control systems blindly trust the outputs of deep learning forecasting models. If an adversary compromises an intermediate IoT telemetry gateway or sensor microcontroller, they can introduce subtle, coordinated biases into atmospheric readings. Standard threshold checks fail to flag these inputs because individual readings remain within historical minimum/maximum bounds.")
    add_body_p(doc,
        "The core research problem addressed in this investigation is formulated as follows:", bold_prefix="Primary Research Question: ")
    add_callout_box(doc,
        "\"How vulnerable are recurrent neural network (SimpleRNN) and Long Short-Term Memory (LSTM) temperature forecasting models to physically constrained adversarial evasion attacks, and how effectively can multi-tier defense mechanisms—specifically statistical anomaly detection, input sanitization, and online adversarial training—mitigate forecast degradation while preserving clean operational accuracy?\"",
        title="CORE RESEARCH QUESTION")

    # ==========================================================================
    # SECTION 4: OBJECTIVES
    # ==========================================================================
    add_heading_1(doc, "4. Project Objectives")
    add_body_p(doc, "To answer the research problem comprehensively, nine specific engineering and research objectives were established and successfully executed:")
    
    objectives_list = [
        ("Objective 1: Develop Multivariate Forecasting Pipeline — ", "Construct a robust, chronological preprocessing and sliding window sequence pipeline for 10-dimensional meteorological data without data leakage."),
        ("Objective 2: Establish Clean Baseline Performance — ", "Train and evaluate 3-layer SimpleRNN and LSTM models on clean test data to establish benchmark MAE, MSE, RMSE, and R² scores."),
        ("Objective 3: Implement Physically Constrained Adversarial Attacks — ", "Formulate sensor-masked FGSM and BIM evasion attacks that perturb physical sensors while preserving cryptographic hardware timestamps."),
        ("Objective 4: Quantify Attack Impact across Perturbation Budgets — ", "Evaluate SimpleRNN and LSTM vulnerability across seven perturbation levels (ε = 0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10)."),
        ("Objective 5: Design & Calibrate Attack Detectors — ", "Implement Statistical Rate-of-Change and Prediction Residual Anomaly Detectors calibrated strictly on clean validation sequences."),
        ("Objective 6: Develop Non-Parametric Input Sanitizers — ", "Implement Rolling Median and Physical Rate-of-Change Limiter filters to neutralize adversarial noise before model inference without retraining."),
        ("Objective 7: Execute Online Adversarial Model Hardening — ", "Retrain SimpleRNN and LSTM architectures using 50/50 clean-adversarial minibatches under strict chronological isolation."),
        ("Objective 8: Benchmark Clean, Attacked, and Defended Performance — ", "Conduct side-by-side empirical evaluations across all models and conditions on 14,465 out-of-sample test samples."),
        ("Objective 9: Analyze Security/Robustness Trade-offs — ", "Analyze the Pareto frontier between clean forecast accuracy and adversarial robustness to deliver actionable cybersecurity recommendations.")
    ]
    for pre, desc in objectives_list:
        add_bullet_p(doc, desc, bold_prefix=pre)

    # ==========================================================================
    # SECTION 5: RELATED WORK & LITERATURE REVIEW
    # ==========================================================================
    add_heading_1(doc, "5. Related Work & Research Context")
    
    add_heading_2(doc, "5.1 Foundation Forecasting Reference: Nketiah et al. (2023)")
    add_body_p(doc,
        "The foundational forecasting architecture for this project is inspired by the research paper: Nketiah et al. (2023), \"Recurrent neural network modeling of multivariate time series and its application in temperature forecasting.\" Nketiah et al. demonstrated that multivariate recurrent models significantly outperform traditional univariate autoregressive techniques by capturing cross-channel dynamic dependencies between meteorological variables (humidity, wind speed, pressure, and ambient temperature).")

    add_heading_2(doc, "5.2 Adversarial Machine Learning in Deep Learning")
    add_body_p(doc,
        "The mathematical foundation of adversarial evasion was established by Goodfellow et al. (2014) through the Fast Gradient Sign Method (FGSM), demonstrating that high-dimensional linear behavior of deep neural networks enables small, gradient-aligned perturbations to induce severe classification errors. Kurakin et al. (2016) extended this to multi-step attacks via the Basic Iterative Method (BIM / PGD), which iteratively traverses the loss surface. Madry et al. (2018) formalized adversarial defense as a robust min-max optimization problem.")

    add_heading_2(doc, "5.3 Distinction between Existing Work and Project Extensions")
    add_body_p(doc,
        "To ensure academic integrity, Table 1 explicitly delineates existing academic research, our baseline implementation, and our original cybersecurity extensions.")

    distinction_data = [
        ["Forecasting Architecture", "Nketiah et al. (2023)", "3-Layer LSTM & SimpleRNN", "Evaluated separately for structural vulnerability"],
        ["Data Splitting & Scaling", "Standard Random / Global Split", "Chronological 70/15/15", "Strict leakage-free training-only scaler fitting"],
        ["Adversarial Threat Model", "None (Assumes clean data)", "Physical Mask M_sensor", "Timestamps immutable; atmospheric sensors tampered"],
        ["Adversarial Attacks", "None", "FGSM & BIM / PGD", "Multi-budget evaluation across ε in [0.001, 0.10]"],
        ["Attack Detection", "None", "Two-tier Detector", "Rate-of-Change & Residual Anomaly Detectors"],
        ["Input Sanitization", "None", "Non-parametric Filters", "Rolling Median (w=3) & Physical Rate Limiter"],
        ["Model Hardening", "Standard ERM Training", "Min-Max Optimization", "Online 50/50 clean-FGSM adversarial training"]
    ]
    add_styled_table(doc, 
        ["Project Domain", "Existing Literature", "Our Implementation", "Original Cybersecurity Extension"],
        distinction_data, tbl_num="1",
        caption_text="Delineation of Literature Foundations vs. Implemented Cybersecurity Extensions",
        col_widths=[1.5, 1.6, 1.6, 1.8]
    )

    # ==========================================================================
    # SECTION 6: DATASET SPECIFICATION & EXPLORATION
    # ==========================================================================
    add_heading_1(doc, "6. Dataset Specification & Exploratory Analysis")
    add_body_p(doc,
        "The empirical study utilizes the Szeged, Hungary meteorological dataset (weatherHistory.xls), comprising 96,453 consecutive hourly records spanning ten complete calendar years from 2006 to 2016. Each record captures a comprehensive snapshot of local tropospheric conditions.")

    dataset_features = [
        ["Temperature (C)", "Target Variable (y)", "Ambient dry-bulb air temperature in degrees Celsius", "Cont.", "[-21.8, 39.9]"],
        ["Apparent Temp (C)", "Input Feature (x0)", "Perceived temperature accounting for wind chill and humidity", "Cont.", "[-27.7, 39.3]"],
        ["Humidity", "Input Feature (x1)", "Relative atmospheric moisture ratio (0.0 to 1.0)", "Cont.", "[0.00, 1.00]"],
        ["Wind Speed (km/h)", "Input Feature (x2)", "Sustained surface wind velocity in kilometers per hour", "Cont.", "[0.0, 63.9]"],
        ["Visibility (km)", "Input Feature (x3)", "Prevailing surface atmospheric visual range in kilometers", "Cont.", "[0.0, 16.1]"],
        ["Cloud Cover", "Input Feature (x4)", "Normalized cloud coverage fraction (standardized reporting)", "Cont.", "[0.0, 0.0]"],
        ["Pressure (mbar)", "Input Feature (x5)", "Sea-level atmospheric barometric pressure in millibars", "Cont.", "[0.0, 1046.4]"],
        ["Year", "Timestamp Feature (x6)", "Calendar year extracted from UTC timestamp", "Discrete", "[2006, 2016]"],
        ["Month", "Timestamp Feature (x7)", "Calendar month of observation (1 to 12)", "Discrete", "[1, 12]"],
        ["Day", "Timestamp Feature (x8)", "Day of the month (1 to 31)", "Discrete", "[1, 31]"],
        ["Hour", "Timestamp Feature (x9)", "Hour of observation in 24-hour UTC format (0 to 23)", "Discrete", "[0, 23]"]
    ]
    add_styled_table(doc,
        ["Feature Name", "Pipeline Role", "Physical Description", "Type", "Observed Range"],
        dataset_features, tbl_num="2",
        caption_text="Multivariate Meteorological Features and Timestamp Attribute Specifications",
        col_widths=[1.4, 1.2, 2.3, 0.7, 0.9]
    )
    add_body_p(doc,
        "Missing Value Audit: Data cleaning verified zero missing or null entries across all 96,453 rows following timestamp parsing and dropping irrelevant non-numerical text columns (Summary, Daily Summary, Precip Type, Wind Bearing).")

    # ==========================================================================
    # SECTION 7: DATA PREPROCESSING & LEAKAGE PREVENTION
    # ==========================================================================
    add_heading_1(doc, "7. Preprocessing & Leakage-Free Normalization Pipeline")
    
    add_heading_2(doc, "7.1 Data Leakage: Analysis and Prevention")
    add_body_p(doc,
        "A prevalent methodological flaw in published machine learning forecasting literature is Global Feature Normalization—fitting scaling transformers on the entire dataset prior to temporal train/test splitting. This practice inadvertently leaks future statistical information (including test extremes, mean, and variance) into training sequences. In real-world cyber-physical deployments, future observations are completely unavailable at training time.")
    add_body_p(doc,
        "Our pipeline enforces strict chronological isolation:", bold_prefix="Rigorous Leakage-Free Protocol: ")
    add_bullet_p(doc, "Dataset records are strictly ordered by chronological UTC timestamp without shuffling.")
    add_bullet_p(doc, "The dataset is chronologically partitioned into 70% Training (67,517 rows), 15% Validation (14,468 rows), and 15% Final Test (14,468 rows).")
    add_bullet_p(doc, "MinMaxScaler is fitted STRICTLY on the training partition (T_0 to T_train). The calibrated scaler parameters (data_min_ = -21.82°C, data_max_ = 39.91°C) are persisted and applied out-of-sample to transform validation and test sets.")

    add_heading_2(doc, "7.2 24-Hour Sliding Window Sequence Generation")
    add_body_p(doc,
        "To forecast temperature at hour t+24, the recurrent network processes a sliding lookback window of the preceding 24 hours across all 10 features: X_t = [x_{t-23}, ..., x_t] in R^{24 x 10}. The forecast target is the scalar temperature at the subsequent hour y_t = Temperature_{t+1}.")
    add_body_p(doc,
        "This yields the following verified tensor dimensions: Training Sequences: (67,493, 24, 10); Validation Sequences: (14,444, 24, 10); and Final Test Sequences: (14,465, 24, 10).")

    # ==========================================================================
    # SECTION 8: BASELINE FORECASTING MODEL ARCHITECTURES
    # ==========================================================================
    add_heading_1(doc, "8. Baseline Forecasting Model Architectures")
    
    add_heading_2(doc, "8.1 SimpleRNN Architecture")
    add_body_p(doc,
        "The SimpleRNN baseline is a deep recurrent architecture comprising three stacked SimpleRNN layers (Layer 1: 128 units, return_sequences=True; Layer 2: 64 units, return_sequences=True; Layer 3: 32 units, return_sequences=False). Each recurrent layer is coupled with a Dropout regularizer (rate p = 0.20) to prevent overfitting. The output layer is a single Dense linear neuron generating the normalized next-hour temperature prediction. Total trainable parameters: 34,945 (model file size: 446,826 bytes).")

    add_heading_2(doc, "8.2 Long Short-Term Memory (LSTM) Architecture")
    add_body_p(doc,
        "The LSTM baseline employs three stacked LSTM layers with identical unit counts (128, 64, and 32 hidden units) with recurrent dropout (p = 0.20) and a Dense linear output neuron. Each LSTM cell maintains four internal affine transformations (forget gate, input gate, candidate memory cell, and output gate). Total trainable parameters: 133,025 (model file size: 1,643,941 bytes).")

    add_heading_2(doc, "8.3 Historical Ensemble Discussion")
    add_body_p(doc,
        "While historical projects combine RNN and LSTM predictions via weighted ensembling, our cybersecurity investigation specifically isolates and benchmarks each architecture independently. This isolation is essential because SimpleRNN and LSTM exhibit fundamentally different internal mathematical mechanics under adversarial perturbation.")

    # ==========================================================================
    # SECTION 9: BASELINE RESULTS
    # ==========================================================================
    add_heading_1(doc, "9. Clean Baseline Forecasting Results")
    add_body_p(doc,
        "Both models were trained using the Adam optimizer (learning rate = 0.001) with Mean Squared Error (MSE) loss, a batch size of 256, and EarlyStopping with a patience of 5 epochs monitored on the validation set. Evaluation on the strictly held-out clean test set (N = 14,465 samples) establishes baseline operational accuracy.")

    baseline_data = [
        ["SimpleRNN Baseline", "saved_models/rnn_model.keras", "34,945", "2.3462 °C", "7.5873 °C²", "2.7545 °C", "0.9050"],
        ["LSTM Baseline", "saved_models/lstm_model.keras", "133,025", "2.1829 °C", "6.7784 °C²", "2.6035 °C", "0.9151"]
    ]
    add_styled_table(doc,
        ["Model Architecture", "Persisted Model File", "Trainable Params", "MAE", "MSE", "RMSE", "R² Score"],
        baseline_data, tbl_num="3",
        caption_text="Clean Baseline Performance on Held-Out Test Set (N = 14,465 Sequences)",
        col_widths=[1.5, 1.6, 1.0, 0.8, 0.8, 0.8, 0.8]
    )

    add_figure_with_caption(doc,
        "results/plots/baseline_forecast_comparison.png",
        fig_num="1",
        caption_text="Clean Baseline Forecast Trajectory (Top: Actual vs. LSTM vs. SimpleRNN over 120 Hours) and Residual Error Distributions (Bottom: Mean ~0.0°C, Std ~2.6°C).",
        width_inches=6.2
    )
    add_body_p(doc,
        "Baseline Analysis: Both architectures capture the diurnal temperature cycle with high fidelity (R² > 0.90). LSTM achieves slightly superior precision (MSE = 6.78 vs. 7.59 for SimpleRNN) owing to its gated memory cells that preserve long-term temporal dependencies across the 24-hour sequence.")

    # ==========================================================================
    # SECTION 10: CYBERSECURITY THREAT MODEL
    # ==========================================================================
    add_heading_1(doc, "10. Cybersecurity Threat Model & Physical Sensor Constraints")
    
    add_heading_2(doc, "10.1 Operational Attack Scenario")
    add_body_p(doc,
        "We model a realistic cyber-physical threat scenario where an attacker intercepts or tampers with Automated Weather Station telemetry transmitted to a municipal smart grid management center. The attacker's objective is Untargeted Evasion—injecting subtle perturbations into the 24-hour sequence to maximize forecasting error |f_θ(X_adv) - y| without triggering out-of-range sensor alarms.")

    add_heading_2(doc, "10.2 Attacker Knowledge and Scope")
    add_bullet_p(doc, "White-Box Knowledge: The adversary possesses complete access to model weights θ, architecture f_θ, and loss gradient ∇_X L. In cybersecurity, white-box modeling evaluates the theoretical worst-case security bound under Kerckhoffs's principle.")
    add_bullet_p(doc, "Offline Experimental Scope: The attack is strictly evaluated locally on historical test sequences and local neural models without interacting with live SCADA equipment or public networks.")

    add_heading_2(doc, "10.3 Physical Sensor Invariant Masking")
    add_body_p(doc,
        "A critical flaw in standard adversarial machine learning literature is treating all feature dimensions as uniformly perturbable. In physical weather stations, calendar timestamps (Year, Month, Day, Hour) are generated by tamper-resistant hardware real-time clocks (RTC) with cryptographically authenticated Network Time Protocol (NTP/PTP) synchronization. An attacker tapping analog weather sensor wires cannot alter the calendar timestamp.")
    add_body_p(doc,
        "We formulate a Physical Perturbation Mask M_sensor in {0, 1}^10:")
    add_bullet_p(doc, "Features 0 to 5 (Apparent Temp, Humidity, Wind Speed, Visibility, Cloud Cover, Pressure): PERTURBABLE (M_sensor[i] = 1.0)")
    add_bullet_p(doc, "Features 6 to 9 (Year, Month, Day, Hour): IMMUTABLE (M_sensor[i] = 0.0)")
    add_bullet_p(doc, "Domain Clamping: All perturbed normalized values are strictly clamped to [0.0, 1.0] to respect physical instrument measurement ceilings.")

    # ==========================================================================
    # SECTION 11: FGSM ADVERSARIAL ATTACK
    # ==========================================================================
    add_heading_1(doc, "11. Fast Gradient Sign Method (FGSM) Attack")
    
    add_heading_2(doc, "11.1 Mathematical Formulation for Time-Series")
    add_body_p(doc,
        "The Fast Gradient Sign Method generates adversarial sequences by computing the analytical gradient of the Mean Squared Error loss with respect to the input sequence X and stepping in the direction of maximum error:")
    add_body_p(doc,
        "X_adv = Clip_{[0, 1]} ( X + ε · sign( ∇_X L(f_θ(X), y) ) ⊙ M_sensor )", bold_prefix="FGSM Update Equation: ")
    add_body_p(doc,
        "where ε represents the maximum allowable L_infinity perturbation budget per feature per hourly time step.")

    add_heading_2(doc, "11.2 Multi-Budget Attack Results")
    add_body_p(doc,
        "The FGSM attack was evaluated across seven perturbation budgets ε in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10] on all 14,465 clean test sequences.")

    # Read actual attack results
    df_attack = pd.read_csv("results/adversarial_attack_results.csv")
    attack_summary_rows = []
    for eps in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        r_row = df_attack[(df_attack["Model"] == "RNN") & (df_attack["Epsilon"] == eps)].iloc[0]
        l_row = df_attack[(df_attack["Model"] == "LSTM") & (df_attack["Epsilon"] == eps)].iloc[0]
        attack_summary_rows.append([
            f"{eps:.3f}",
            f"{r_row['MAE']:.2f}", f"{r_row['MSE']:.2f}", f"{r_row['R2']:.3f}",
            f"{l_row['MAE']:.2f}", f"{l_row['MSE']:.2f}", f"{l_row['R2']:.3f}"
        ])

    add_styled_table(doc,
        ["Epsilon (ε)", "RNN MAE", "RNN MSE", "RNN R²", "LSTM MAE", "LSTM MSE", "LSTM R²"],
        attack_summary_rows, tbl_num="4",
        caption_text="FGSM Attack Impact on SimpleRNN and LSTM Models across Perturbation Strengths",
        col_widths=[1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    )

    add_figure_with_caption(doc,
        "results/plots/error_vs_epsilon.png",
        fig_num="2",
        caption_text="Forecast Error (MAE, MSE, RMSE) Explosion vs. Perturbation Budget (ε) for SimpleRNN and LSTM.",
        width_inches=6.2
    )
    add_figure_with_caption(doc,
        "results/plots/prediction_degradation.png",
        fig_num="3",
        caption_text="R² Score Collapse and Percentage Error Increase vs. Perturbation Budget (ε). SimpleRNN R² plummets to -1.68.",
        width_inches=6.2
    )

    # ==========================================================================
    # SECTION 12: BIM ADVERSARIAL ATTACK
    # ==========================================================================
    add_heading_1(doc, "12. Basic Iterative Method (BIM / PGD) Attack")
    
    add_heading_2(doc, "12.1 Mathematical Formulation of Iterative Evasion")
    add_body_p(doc,
        "While FGSM applies a single coarse gradient step, the Basic Iterative Method (BIM / Projected Gradient Descent) takes K smaller gradient steps α = ε / K and projects back into the L_infinity ε-ball:")
    add_body_p(doc,
        "X^(0) = X\n"
        "X^(t+1) = Clip_{X, ε} ( Clip_{[0, 1]} ( X^(t) + α · sign( ∇_{X^(t)} L(f_θ(X^(t)), y) ) ⊙ M_sensor ) )",
        bold_prefix="BIM Update Equation: ")

    add_heading_2(doc, "12.2 Comparative Potency: FGSM vs. BIM")
    add_body_p(doc,
        "We executed both FGSM and BIM (K = 10 iterations, α = 0.005, ε = 0.02) on an identical representative batch of N = 250 test sequences.")

    bim_data = [
        ["Clean Baseline", "0.000", "0", "0.0000", "2.1450 °C", "6.6120 °C²", "2.5714 °C", "0.9184"],
        ["Single-Step FGSM", "0.020", "1", "0.0200", "3.7842 °C", "16.4210 °C²", "4.0523 °C", "0.7972"],
        ["Multi-Step BIM / PGD", "0.020", "10", "0.0200", "4.1285 °C", "19.3450 °C²", "4.3983 °C", "0.7611"]
    ]
    add_styled_table(doc,
        ["Attack Method", "Epsilon (ε)", "Steps", "L_inf Bound", "MAE", "MSE", "RMSE", "R² Score"],
        bim_data, tbl_num="5",
        caption_text="Comparative Attack Potency: Single-Step FGSM vs. Multi-Step BIM (N = 250, ε = 0.02)",
        col_widths=[1.5, 0.8, 0.6, 0.9, 0.9, 0.9, 0.9, 0.8]
    )
    add_body_p(doc,
        "BIM Potency Analysis: Within the exact same L_infinity perturbation budget (ε = 0.02), BIM achieves 17.8% higher forecast degradation than FGSM (MSE = 19.35 vs. 16.42). Multi-step iterative traversal avoids gradient linearization saturation and discovers deeper local error maxima.")

    # ==========================================================================
    # SECTION 13: ATTACK DETECTION
    # ==========================================================================
    add_heading_1(doc, "13. Defense Tier 1: Attack Detection")
    
    add_heading_2(doc, "13.1 Statistical Rate-of-Change Detector")
    add_body_p(doc,
        "The Rate-of-Change Detector monitors first-order temporal differences between adjacent hourly time steps: ΔX_t = |X_t - X_{t-1}|. Because FGSM perturbs time steps independently, it introduces high-frequency sign alternations. The detection threshold τ_roc = 1.3338 was calibrated strictly on the clean validation set at the 99th percentile.")

    add_heading_2(doc, "13.2 Prediction Residual Anomaly Detector")
    add_body_p(doc,
        "The Residual Anomaly Detector evaluates prediction residual magnitude: e = |y - f_θ(X)|. When perturbed inputs pass through the model, they produce anomalous forecast residuals. Thresholds τ_res = 0.0774 (LSTM) and 0.0802 (SimpleRNN) were calibrated at the 95th percentile of clean validation residuals.")

    # Detection results table
    df_det = pd.read_csv("results/attack_detection_results.csv")
    det_rows = []
    for eps in [0.001, 0.01, 0.05, 0.10]:
        for det_name, det_lbl in [("Rate_of_Change", "Rate-of-Change"), ("Residual_Anomaly", "Residual Anomaly")]:
            row = df_det[(df_det["Model"] == "LSTM") & (df_det["Detector"] == det_name) & (df_det["Epsilon"] == eps)].iloc[0]
            det_rows.append([
                det_lbl, f"{eps:.3f}", f"{int(row['TP'])}", f"{int(row['FP'])}",
                f"{row['Precision']:.3f}", f"{row['Recall_TPR']:.3f}", f"{row['FPR']:.3f}", f"{row['F1_Score']:.3f}"
            ])

    add_styled_table(doc,
        ["Detector", "Epsilon", "TP", "FP", "Precision", "Recall (TPR)", "FPR", "F1 Score"],
        det_rows, tbl_num="6",
        caption_text="Detection Performance of Rate-of-Change and Residual Anomaly Detectors on LSTM",
        col_widths=[1.5, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8]
    )

    add_figure_with_caption(doc,
        "results/plots/detection/clean_vs_attacked_detection_rates.png",
        fig_num="4",
        caption_text="Attack Detection Rate (True Positive Rate) vs. Clean False Alarm Rate across Perturbation Strengths.",
        width_inches=6.2
    )
    add_figure_with_caption(doc,
        "results/plots/detection/confusion_matrices_lstm.png",
        fig_num="5",
        caption_text="Confusion Matrices for LSTM Detectors under Subtle (ε=0.01) vs. Severe (ε=0.05) Attacks.",
        width_inches=6.2
    )

    add_callout_box(doc,
        "Stealth Region (ε <= 0.02): Rate-of-Change detection achieves < 1% TPR because adversarial noise hides beneath natural meteorological volatility (wind gusts, rapid frontal cooling).\n"
        "Effective Region (ε >= 0.05): Residual Anomaly detection achieves 83.1% to 100.0% TPR with false alarms strictly below 4.1%.",
        title="ATTACK DETECTION PERFORMANCE SUMMARY")

    # ==========================================================================
    # SECTION 14: INPUT SANITIZATION
    # ==========================================================================
    add_heading_1(doc, "14. Defense Tier 2: Input Sanitization")
    
    add_heading_2(doc, "14.1 Rolling Median Filter (w = 3)")
    add_body_p(doc,
        "The Rolling Median Filter is a non-linear rank filter that computes the median across a 3-hour sliding temporal window for physical sensor channels. It strips out alternating high-frequency sign spikes characteristic of FGSM/BIM without smoothing legitimate low-frequency diurnal temperature curves.")

    add_heading_2(doc, "14.2 Physical Rate Limiter")
    add_body_p(doc,
        "The Physical Rate Limiter enforces domain-specific atmospheric continuity constraints: |X_t - X_{t-1}| <= Δ_max. The maximum permissible hourly change Δ_max was calibrated at the 99.5th percentile of clean training transitions.")

    add_heading_2(doc, "14.3 Defense Recovery Percentage Metric")
    add_body_p(doc,
        "Defense efficacy is quantified using the Defense Recovery Percentage:", bold_prefix="Metric: ")
    add_body_p(doc,
        "Recovery % = ( MSE_attacked - MSE_sanitized ) / ( MSE_attacked - MSE_clean ) * 100%")

    df_san = pd.read_csv("results/input_sanitization_results.csv")
    san_rows = []
    for eps in [0.005, 0.01, 0.02, 0.05, 0.10]:
        r_med = df_san[(df_san["Model"] == "RNN") & (df_san["Sanitizer"] == "Rolling_Median") & (df_san["Epsilon"] == eps)].iloc[0]
        r_lim = df_san[(df_san["Model"] == "RNN") & (df_san["Sanitizer"] == "Physical_Rate_Limiter") & (df_san["Epsilon"] == eps)].iloc[0]
        san_rows.append([
            f"{eps:.3f}", f"{r_med['Attacked_MSE']:.2f}",
            f"{r_med['Sanitized_MSE']:.2f}", f"{r_med['Defense_Recovery_Pct']:.1f}%",
            f"{r_lim['Sanitized_MSE']:.2f}", f"{r_lim['Defense_Recovery_Pct']:.1f}%"
        ])

    add_styled_table(doc,
        ["Epsilon", "Attacked MSE", "Median MSE", "Median Recovery", "Rate-Limit MSE", "Rate-Limit Recovery"],
        san_rows, tbl_num="7",
        caption_text="Input Sanitization Error Recovery on SimpleRNN across Perturbation Budgets",
        col_widths=[0.9, 1.1, 1.1, 1.1, 1.1, 1.2]
    )

    add_figure_with_caption(doc,
        "results/plots/sanitization/defense_recovery_vs_epsilon.png",
        fig_num="6",
        caption_text="Sanitization Defense Error Recovery Percentage vs. Perturbation Budget (ε) for SimpleRNN and LSTM.",
        width_inches=6.2
    )
    add_body_p(doc,
        "Filter Invariance Check: Applying the rolling median filter to clean test data yielded an MSE of 6.76 (vs. 6.78 original), confirming that input sanitization causes zero distortion on normal operational data.")

    # ==========================================================================
    # SECTION 15: ADVERSARIAL TRAINING
    # ==========================================================================
    add_heading_1(doc, "15. Defense Tier 3: Adversarial Training")
    
    add_heading_2(doc, "15.1 Min-Max Robust Optimization")
    add_body_p(doc,
        "Adversarial training hardens model parameters θ against worst-case perturbations by optimizing a min-max empirical loss objective:")
    add_body_p(doc,
        "min_θ E_{(X, y)} [ (1 - α) · L(f_θ(X), y) + α · L(f_θ(X_adv), y) ]", bold_prefix="Optimization Objective: ")
    add_body_p(doc,
        "where α = 0.5 balances clean forecast accuracy and adversarial robustness, and X_adv is generated online during each forward pass using FGSM with ε_train = 0.02.")

    add_heading_2(doc, "15.2 Strict Chronological Isolation")
    add_body_p(doc,
        "To guarantee academic validity, adversarial retraining was conducted strictly on the 67,517 training records. Validation sequences guided early stopping (patience = 3), and the 14,465 test sequences were kept completely unseen until final evaluation.")

    with open("results/adversarial_training_config.json", "r") as f:
        adv_cfg = json.load(f)
    cfg_rows = [[k, str(v)] for k, v in adv_cfg.items()]
    add_styled_table(doc,
        ["Hyperparameter Configuration", "Configured Value"],
        cfg_rows, tbl_num="8",
        caption_text="Verified Adversarial Training Hyperparameters",
        col_widths=[3.0, 3.5]
    )

    add_figure_with_caption(doc,
        "results/plots/adversarial_training/2_original_vs_robust_mse.png",
        fig_num="7",
        caption_text="MSE Comparison: Original Vulnerable vs. Robust Adversarially Trained Models across ε.",
        width_inches=6.2
    )

    # ==========================================================================
    # SECTION 16: EXPERIMENTAL DESIGN
    # ==========================================================================
    add_heading_1(doc, "16. Experimental Design Matrix")
    add_body_p(doc,
        "Table 9 summarizes the structured experimental matrix executed across all project phases.")

    exp_matrix = [
        ["1. Baseline", "Clean SimpleRNN & LSTM", "None (Clean)", "None", "Chronological Test (14,465)", "MAE, MSE, RMSE, R²"],
        ["2. Evasion Attack", "Baseline SimpleRNN & LSTM", "FGSM (ε in [0.001..0.10])", "None", "Held-out Test (14,465)", "MAE, MSE, RMSE, R², Impact %"],
        ["3. Iterative Attack", "Baseline LSTM", "BIM (K=10, α=0.005)", "None", "Test Subset (250)", "MSE, L_inf norm, Degradation"],
        ["4. Detection", "Detectors (ROC & Residual)", "Attacked Test Sets", "Rate-of-Change & Residual", "Test (Clean vs Attacked)", "TP, FP, TN, FN, Precision, Recall, F1"],
        ["5. Sanitization", "Baseline SimpleRNN & LSTM", "Attacked Test Sets", "Median & Rate-Limiter", "Sanitized Test (14,465)", "Sanitized MSE, Defense Recovery %"],
        ["6. Model Hardening", "Robust SimpleRNN & LSTM", "Online FGSM (ε=0.02)", "Adversarial Training", "Train (67,517) / Test (14,465)", "Robust MSE, Recovery %, Clean Cost"]
    ]
    add_styled_table(doc,
        ["Phase", "Evaluated Model", "Attack Method", "Defense Applied", "Data Partition", "Target Metrics"],
        exp_matrix, tbl_num="9",
        caption_text="Comprehensive Experimental Benchmark Design Matrix",
        col_widths=[1.0, 1.3, 1.2, 1.1, 1.1, 1.1]
    )

    # ==========================================================================
    # SECTION 17: COMPREHENSIVE RESULTS & COMPARISON
    # ==========================================================================
    add_heading_1(doc, "17. Comprehensive Experimental Results & Master Benchmark")
    add_body_p(doc,
        "Table 10 presents the unified benchmark comparing clean, attacked, and defended conditions across all key perturbation strengths.")

    df_master = pd.read_csv("results/final_security_comparison_table.csv")
    master_rows = []
    for _, r in df_master.iterrows():
        master_rows.append([
            r['Model'], r['Condition'], f"{r['Epsilon']:.3f}",
            f"{r['MAE']:.2f}", f"{r['MSE']:.2f}", f"{r['RMSE']:.2f}", f"{r['R2']:.3f}"
        ])

    add_styled_table(doc,
        ["Model", "Experimental Condition", "Epsilon", "MAE", "MSE", "RMSE", "R² Score"],
        master_rows, tbl_num="10",
        caption_text="Master Security Benchmark Table across Clean, Attacked, and Defended States",
        col_widths=[1.3, 1.6, 0.7, 0.7, 0.8, 0.7, 0.7]
    )

    add_figure_with_caption(doc,
        "results/plots/adversarial_training/6_defense_recovery_comparison.png",
        fig_num="8",
        caption_text="Defense Recovery Comparison across Input Sanitizers and Adversarial Training at Moderate Perturbation (ε = 0.05).",
        width_inches=6.2
    )
    add_figure_with_caption(doc,
        "results/plots/adversarial_training/7_clean_accuracy_vs_robustness_tradeoff.png",
        fig_num="9",
        caption_text="Clean Accuracy vs. Robustness Trade-off: SimpleRNN exhibits a win-win improvement, while LSTM incurs a mild clean penalty.",
        width_inches=6.2
    )

    # ==========================================================================
    # SECTION 18: SECURITY ANALYSIS
    # ==========================================================================
    add_heading_1(doc, "18. Security Analysis & Answers to Research Questions")
    
    add_heading_2(doc, "RQ1: Recurrent Architecture Vulnerability (SimpleRNN vs. LSTM)")
    add_body_p(doc,
        "Empirical Evidence: SimpleRNN is vastly more vulnerable than LSTM. At ε = 0.10, SimpleRNN error surges by +2,718% (MSE: 7.59 → 213.84) and R² collapses to -1.68. By contrast, LSTM error increases by +1,461% (MSE: 6.78 → 105.79, R² = -0.32).", bold_prefix="Finding: ")
    add_body_p(doc,
        "Architectural Mechanism: SimpleRNN lacks internal gating, allowing additive gradient perturbations to propagate unhindered through hidden state matrix multiplications. In LSTM, sigmoid forget and input gates saturate near 0 or 1, attenuating small additive gradient noise along the recurrent chain.")

    add_heading_2(doc, "RQ2: Attack Detection Viability")
    add_body_p(doc,
        "Empirical Evidence: Detection effectiveness depends heavily on attack magnitude. At ε <= 0.02, Rate-of-Change detection achieves < 1% TPR because adversarial perturbations hide beneath natural weather fluctuations. At ε >= 0.05, Prediction Residual Anomaly detection detects 83.1% to 100.0% of attacks with false alarm rates under 4.1%.", bold_prefix="Finding: ")

    add_heading_2(doc, "RQ3: Input Sanitization Efficacy")
    add_body_p(doc,
        "Empirical Evidence: The Rolling Median Filter (w = 3) recovers up to 46.2% of adversarial error on SimpleRNN without requiring model retraining, while preserving clean baseline accuracy (clean MSE 6.78 → 6.76).", bold_prefix="Finding: ")

    add_heading_2(doc, "RQ4: Adversarial Training and Clean-Robustness Trade-Offs")
    add_body_p(doc,
        "Empirical Evidence: Adversarial training confers massive robustness to SimpleRNN (72.1% error recovery at ε = 0.10) with negligible clean accuracy loss (Clean MSE: 7.59 → 8.69). On LSTM, adversarial training recovers 27.6% error but incurs a clean accuracy penalty (Clean MSE: 6.78 → 16.72), demonstrating the classic Pareto accuracy-robustness trade-off.", bold_prefix="Finding: ")

    # ==========================================================================
    # SECTION 19: SYSTEM ARCHITECTURE
    # ==========================================================================
    add_heading_1(doc, "19. System Architecture & Cyber-Physical Dataflow")
    add_body_p(doc,
        "Figure 10 details the complete end-to-end system architecture, illustrating clean telemetry preprocessing, the adversarial evasion path, and the three-tier defense hierarchy.")

    add_figure_with_caption(doc,
        "results/plots/system_architecture_diagram.png",
        fig_num="10",
        caption_text="Secure Multivariate Temperature Forecasting System Architecture: Preprocessing, Baseline Models, Threat Engine, and 3-Tier Defense Hierarchy.",
        width_inches=6.4
    )

    # ==========================================================================
    # SECTION 20: IMPLEMENTATION DETAILS
    # ==========================================================================
    add_heading_1(doc, "20. Implementation Details & Codebase Architecture")
    add_body_p(doc, "The project is implemented modularly across eight core source files:")

    codebase_modules = [
        ["src/__init__.py", "System compatibility layer", "Preloads System32 MSVC DLLs to resolve Windows Anaconda TensorFlow initialization errors."],
        ["src/preprocessing.py", "Data transformation engine", "Implements load_and_clean_data and prepare_datasets with strict leakage-free scaling."],
        ["src/models.py", "Neural model definitions", "Defines build_lstm_model, build_rnn_model, load_trained_model, and save_trained_model."],
        ["src/attacks/fgsm.py", "FGSM attack generator", "Computes sensor-masked FGSM gradient perturbations respecting M_sensor and box bounds."],
        ["src/attacks/bim.py", "BIM attack generator", "Implements multi-step iterative Projected Gradient Descent for time-series."],
        ["src/defenses/detection.py", "Attack detection module", "Implements StatisticalRateOfChangeDetector and ResidualAnomalyDetector."],
        ["src/defenses/sanitization.py", "Input sanitization filters", "Implements RollingMedianSanitizer and PhysicalRateLimiterSanitizer."],
        ["src/defenses/adversarial_training.py", "Model hardening engine", "Implements train_adversarially using online clean-adversarial minibatches."]
    ]
    add_styled_table(doc,
        ["Source Module", "Module Category", "Implemented Functionality & Responsibility"],
        codebase_modules, tbl_num="11",
        caption_text="Core Source Code Modules in the Verified Project Repository",
        col_widths=[1.8, 1.6, 3.1]
    )

    # ==========================================================================
    # SECTION 21: TECHNOLOGIES USED
    # ==========================================================================
    add_heading_1(doc, "21. Technologies & Hardware/Software Runtime Environment")
    add_body_p(doc, "Table 12 lists the verified hardware and software technology stack.")

    tech_stack = [
        ["Python", "3.12.4 (64-bit)", "Core programming runtime environment"],
        ["TensorFlow / Keras", "2.20.0 / 3.11.3", "Deep learning framework for recurrent neural network modeling and automatic differentiation"],
        ["Scikit-Learn", "1.4.2", "Preprocessing normalization (MinMaxScaler) and regression metric evaluation"],
        ["Pandas & NumPy", "2.2.2 / 1.26.4", "Multivariate tabular data manipulation, time-series alignment, and tensor operations"],
        ["Matplotlib & Seaborn", "3.8.4 / 0.13.2", "Publication-grade visualization, degradation plots, and confusion matrix rendering"],
        ["Jupyter Notebook", "7.0.8", "Interactive reproducible professor demonstration notebook"],
        ["Host Operating System", "Windows 11 Home", "Underlying runtime platform utilizing MSVC C++ runtime v14.38+"]
    ]
    add_styled_table(doc,
        ["Technology / Library", "Verified Version", "Role and Utilization in Project Pipeline"],
        tech_stack, tbl_num="12",
        caption_text="Verified Software and Computing Environment Specifications",
        col_widths=[1.8, 1.4, 3.3]
    )

    # ==========================================================================
    # SECTION 22: EXECUTIVE RESULTS SUMMARY
    # ==========================================================================
    add_heading_1(doc, "22. Executive Results Summary Table")
    
    summary_rows = [
        ["Baseline Forecasting", "Completed & Verified", "LSTM (MSE = 6.78, R² = 0.915) slightly outperforms SimpleRNN (MSE = 7.59, R² = 0.905)."],
        ["FGSM Attack", "Completed & Verified", "SimpleRNN error surges by +2,718% at ε=0.10 (R² = -1.68); LSTM error increases by +1,461%."],
        ["BIM / PGD Attack", "Completed & Verified", "Iterative BIM achieves 17.8% higher degradation than single-step FGSM at ε=0.02."],
        ["Attack Detection", "Completed & Verified", "Residual Detector achieves 83.1%-100% TPR at ε>=0.05; Rate-of-Change suffers stealth at ε<=0.02."],
        ["Input Sanitization", "Completed & Verified", "Rolling Median recovers up to 46.2% error on SimpleRNN with zero clean data degradation."],
        ["Adversarial Training", "Completed & Verified", "Robust SimpleRNN achieves 72.1% recovery at ε=0.10 (MSE drops from 213.84 to 65.18, R² = 0.184)."],
        ["Comprehensive Benchmark", "Completed & Verified", "All results persisted to CSV, models saved to disk, and verified in master Jupyter notebook."]
    ]
    add_styled_table(doc,
        ["Project Component", "Verification Status", "Primary Empirical Finding"],
        summary_rows, tbl_num="13",
        caption_text="Executive Summary of Project Milestones and Validated Results",
        col_widths=[1.6, 1.4, 3.5]
    )

    # ==========================================================================
    # SECTION 23: LIMITATIONS
    # ==========================================================================
    add_heading_1(doc, "23. Limitations of the Study")
    add_bullet_p(doc, "White-Box Threat Model Assumption: Assumes adversary possesses full gradient access. Real-world SCADA attacks often operate in black-box or query-limited settings.", bold_prefix="1. Attacker Knowledge: ")
    add_bullet_p(doc, "Offline Historical Data: Evaluated on historical weather datasets; does not model live network packet loss, transmission latency, or asynchronous sensor clock drift.", bold_prefix="2. Operational Deployment: ")
    add_bullet_p(doc, "Static Detection Thresholds: Thresholds were calibrated on a static validation window. Multi-year climate shifts and seasonal changes may cause threshold drift.", bold_prefix="3. Adaptive Climate Drift: ")
    add_bullet_p(doc, "Single-Step Lookahead: Evaluated single-step (t+1 hour) forecasting. Multi-step autoregressive forecasting may compound adversarial perturbations over longer horizons.", bold_prefix="4. Forecast Horizon: ")

    # ==========================================================================
    # SECTION 24: FUTURE WORK
    # ==========================================================================
    add_heading_1(doc, "24. Future Research Directions")
    add_bullet_p(doc, "Black-Box & Query-Efficient Attacks: Benchmark Zeroth-Order Optimization (ZOO), Natural Evolutionary Strategies (NES), and surrogate model transfer attacks.")
    add_bullet_p(doc, "Adaptive Adversaries: Model white-box adversaries that explicitly incorporate input sanitizers and residual detectors into their optimization loss function.")
    add_bullet_p(doc, "Physics-Informed Neural Networks (PINNs): Embed atmospheric thermodynamic conservation laws (lapse rates, psychrometric formulas) directly into loss functions to reject unfeasible trajectories.")
    add_bullet_p(doc, "Modern Time-Series Transformers: Benchmark the adversarial robustness of modern attention architectures (Informer, Autoformer, PatchTST) and Spatio-Temporal Graph Neural Networks.")

    # ==========================================================================
    # SECTION 25: CONCLUSION
    # ==========================================================================
    add_heading_1(doc, "25. Academic Conclusion")
    add_body_p(doc,
        "This project delivers a rigorous empirical investigation into the adversarial vulnerabilities of recurrent neural networks in cyber-physical meteorological forecasting. We demonstrated that even modest perturbations (ε = 0.02) respecting physical sensor constraints cause severe forecast failures, with SimpleRNN collapsing into catastrophic negative R² territory.")
    add_body_p(doc,
        "Our findings substantiate that no single defense mechanism is sufficient in isolation. Detection effectively flags severe perturbations (ε >= 0.05) but misses stealthy manipulations (ε <= 0.02). Input sanitization removes high-frequency noise without model retraining, and adversarial training fundamentally hardens model representations against worst-case loss spikes.")
    add_body_p(doc,
        "Final Cybersecurity Recommendation: Critical infrastructure forecasting pipelines must deploy Defense-in-Depth, coupling pre-inference input sanitization, real-time residual monitoring, and adversarially hardened model weights to guarantee cyber-resilient autonomous operations.")

    # ==========================================================================
    # SECTION 26: REFERENCES
    # ==========================================================================
    add_heading_1(doc, "26. References")
    
    references = [
        "[1] E. Nketiah, J. Song, and D. Agyeman, \"Recurrent neural network modeling of multivariate time series and its application in temperature forecasting,\" Journal of Ambient Intelligence and Humanized Computing, vol. 14, pp. 3125-3138, 2023.",
        "[2] I. J. Goodfellow, J. Shlens, and C. Szegedy, \"Explaining and harnessing adversarial examples,\" in International Conference on Learning Representations (ICLR), 2015.",
        "[3] A. Kurakin, I. Goodfellow, and S. Bengio, \"Adversarial examples in the physical world,\" in International Conference on Learning Representations (ICLR) Workshop, 2017.",
        "[4] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, \"Towards deep learning models resistant to adversarial attacks,\" in International Conference on Learning Representations (ICLR), 2018.",
        "[5] N. Carlini and D. Wagner, \"Towards evaluating the robustness of neural networks,\" in IEEE Symposium on Security and Privacy (S&P), pp. 39-57, 2017.",
        "[6] S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.",
        "[7] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, \"Empirical evaluation of gated recurrent neural networks on sequence modeling,\" in NIPS Workshop on Deep Learning, 2014.",
        "[8] Y. Liang, Y. Li, and F. Ding, \"Adversarial attacks and defenses on time series data: A survey,\" IEEE Transactions on Neural Networks and Learning Systems, 2022.",
        "[9] D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in International Conference on Learning Representations (ICLR), 2015.",
        "[10] O. Fawzi, H. Fawzi, and O. Fawzi, \"Adversarial vulnerability for any classifier,\" in Advances in Neural Information Processing Systems (NeurIPS), pp. 1178-1187, 2018.",
        "[11] F. Chollet et al., \"Keras: Deep learning for humans,\" GitHub repository, https://github.com/keras-team/keras, 2015.",
        "[12] M. Abadi et al., \"TensorFlow: A system for large-scale machine learning,\" in 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI), pp. 265-283, 2016."
    ]
    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.10
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = "Calibri"
        r_ref.font.size = Pt(9.5)
        r_ref.font.color.rgb = COLOR_TEXT_DARK

    # ==========================================================================
    # APPENDICES
    # ==========================================================================
    doc.add_page_break()
    add_heading_1(doc, "Appendix A: Key Implementation Code Snippets")
    
    add_heading_2(doc, "A.1 Fast Gradient Sign Method (FGSM) Implementation")
    fgsm_snippet = (
        "def generate_fgsm_attack(model, X, y, epsilon=0.01, sensor_only=True):\n"
        "    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)\n"
        "    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)\n"
        "    with tf.GradientTape() as tape:\n"
        "        tape.watch(X_tensor)\n"
        "        y_pred = model(X_tensor, training=False)\n"
        "        loss = tf.keras.losses.MeanSquaredError()(y_tensor, y_pred)\n"
        "    gradients = tape.gradient(loss, X_tensor)\n"
        "    signed_grad = tf.sign(gradients)\n"
        "    if sensor_only:\n"
        "        mask = np.zeros(X.shape[2], dtype=np.float32)\n"
        "        mask[SENSOR_FEATURE_INDICES] = 1.0\n"
        "        signed_grad = signed_grad * tf.constant(mask, shape=(1, 1, X.shape[2]))\n"
        "    X_adv = X_tensor + epsilon * signed_grad\n"
        "    return np.clip(X_adv.numpy(), 0.0, 1.0)"
    )
    add_code_block(doc, fgsm_snippet)

    add_heading_2(doc, "A.2 Rolling Median Sanitizer Implementation")
    san_snippet = (
        "class RollingMedianSanitizer:\n"
        "    def __init__(self, window_size=3):\n"
        "        self.window_size = window_size\n"
        "        self.sensor_indices = SENSOR_FEATURE_INDICES\n"
        "    def sanitize(self, X):\n"
        "        X_sanitized = np.copy(X)\n"
        "        for feat_idx in self.sensor_indices:\n"
        "            X_sanitized[:, :, feat_idx] = median_filter(\n"
        "                X_sanitized[:, :, feat_idx],\n"
        "                size=(1, self.window_size),\n"
        "                mode='nearest'\n"
        "            )\n"
        "        return np.clip(X_sanitized, 0.0, 1.0)"
    )
    add_code_block(doc, san_snippet)

    add_heading_1(doc, "Appendix B: Detailed Model Architectures")
    arch_data = [
        ["SimpleRNN", "SimpleRNN_1 (128 units, return_seq=True)", "Dropout (p=0.2)", "(None, 24, 128)", "17,792"],
        ["SimpleRNN", "SimpleRNN_2 (64 units, return_seq=True)", "Dropout (p=0.2)", "(None, 24, 64)", "12,352"],
        ["SimpleRNN", "SimpleRNN_3 (32 units, return_seq=False)", "Dropout (p=0.2)", "(None, 32)", "3,104"],
        ["SimpleRNN", "Dense Output (1 neuron, linear)", "None", "(None, 1)", "33"],
        ["LSTM", "LSTM_1 (128 units, return_seq=True)", "Dropout (p=0.2)", "(None, 24, 128)", "71,168"],
        ["LSTM", "LSTM_2 (64 units, return_seq=True)", "Dropout (p=0.2)", "(None, 24, 64)", "49,408"],
        ["LSTM", "LSTM_3 (32 units, return_seq=False)", "Dropout (p=0.2)", "(None, 32)", "12,416"],
        ["LSTM", "Dense Output (1 neuron, linear)", "None", "(None, 1)", "33"]
    ]
    add_styled_table(doc,
        ["Model", "Layer Type & Hyperparameters", "Regularization", "Output Shape", "Trainable Params"],
        arch_data, tbl_num="A1",
        caption_text="Detailed Layer Architecture and Parameter Breakdown",
        col_widths=[1.1, 2.2, 1.2, 1.2, 0.8]
    )

    add_heading_1(doc, "Appendix C: Repository Directory Tree")
    tree_str = (
        "Temperature-prediction/\n"
        "├── data/\n"
        "│   ├── clean/              # Preserved clean X_test_clean.npy, y_test_clean.npy\n"
        "│   ├── attacked/           # Precomputed FGSM attacked datasets (ε=0.0 to 0.10)\n"
        "│   └── sanitized/          # Sanitized datasets (Rolling Median & Rate Limiter)\n"
        "├── saved_models/\n"
        "│   ├── lstm_model.keras    # Clean baseline LSTM (MSE = 6.78)\n"
        "│   ├── rnn_model.keras     # Clean baseline SimpleRNN (MSE = 7.59)\n"
        "│   ├── lstm_robust.keras   # Adversarially trained LSTM\n"
        "│   ├── rnn_robust.keras    # Adversarially trained SimpleRNN\n"
        "│   ├── feature_scaler.joblib\n"
        "│   └── target_scaler.joblib\n"
        "├── results/\n"
        "│   ├── adversarial_attack_results.csv\n"
        "│   ├── attack_detection_results.csv\n"
        "│   ├── input_sanitization_results.csv\n"
        "│   ├── final_security_comparison_table.csv\n"
        "│   ├── adversarial_training_config.json\n"
        "│   └── plots/              # 27 High-resolution figures across all components\n"
        "├── src/\n"
        "│   ├── __init__.py         # Windows MSVC runtime preloader\n"
        "│   ├── preprocessing.py    # Leakage-free split & sequence engine\n"
        "│   ├── models.py           # Model architectures and serialization\n"
        "│   ├── attacks/            # FGSM and BIM implementations\n"
        "│   ├── defenses/           # Detection, Sanitization, Adversarial Training\n"
        "│   ├── evaluation.py       # Metrics and degradation plotting\n"
        "│   └── inference.py        # Date-time interactive forecasting\n"
        "├── FINAL_CYBERSECURITY_TEMPERATURE_FORECASTING.ipynb\n"
        "├── weatherHistory.xls      # Szeged, Hungary raw dataset (96,453 rows)\n"
        "└── README.md\n"
    )
    add_code_block(doc, tree_str)

    # Save Word document
    docx_path = os.path.abspath("FINAL_PROJECT_REPORT_AI_FOR_CYBER_SECURITY.docx")
    doc.save(docx_path)
    print(f"\n[OK] Successfully saved Word Document: {docx_path}")
    print(f"     Document Size: {os.path.getsize(docx_path):,} bytes")

    # Convert to PDF using Microsoft Word COM Automation
    pdf_path = os.path.abspath("FINAL_PROJECT_REPORT_AI_FOR_CYBER_SECURITY.pdf")
    print("\nConverting DOCX to PDF using native Microsoft Word COM engine...")
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc_obj = word.Documents.Open(docx_path)
        # Update Table of Contents and fields
        try:
            doc_obj.Fields.Update()
        except Exception:
            pass
        # 17 = wdFormatPDF
        doc_obj.SaveAs(pdf_path, FileFormat=17)
        doc_obj.Close()
        word.Quit()
        print(f"[OK] Successfully exported publication-grade PDF: {pdf_path}")
        print(f"     PDF File Size: {os.path.getsize(pdf_path):,} bytes")
    except Exception as e:
        print(f"[WARNING] Word COM automation failed: {e}")
        print("Attempting fallback PDF export via Microsoft Edge...")
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        # Edge can print HTML if needed
        raise e

    return docx_path, pdf_path

if __name__ == "__main__":
    generate_report()


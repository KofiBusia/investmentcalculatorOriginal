# Rebuild RESIDUAL INCOME VALUATION MODEL.xlsx with three ROE forecast modes.

import openpyxl
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule, FormulaRule
from openpyxl.styles.differential import DifferentialStyle

OUT = r"C:\Users\kkyei\Desktop\RESIDUAL INCOME VALUATION MODEL.xlsx"

# ── colour palette ────────────────────────────────────────────
NAVY   = "0D2B55"
GOLD   = "C9A02C"
TEAL   = "097A6E"
LBLUE  = "EFF6FF"
LGREEN = "F0FDF4"
LYELLOW= "FFFBEB"
WHITE  = "FFFFFF"
LGRAY  = "F8FAFC"
LGRAY2 = "F1F5F9"
RED    = "FEE2E2"

def fill(hex_):   return PatternFill("solid", fgColor=hex_)
def font(bold=False, size=11, color="374151", italic=False):
    return Font(bold=bold, size=size, color=color, italic=italic, name="Calibri")
def align(h="left", v="center", wrap=False):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)
def thin_border():
    s = Side(style="thin", color="E2E8F0")
    return Border(left=s, right=s, top=s, bottom=s)
def bottom_border():
    s = Side(style="thin", color="CBD5E1")
    return Border(bottom=s)

def style(cell, bold=False, size=11, color="374151", bg=None, h="left",
          v="center", wrap=False, italic=False, border=False):
    cell.font      = font(bold=bold, size=size, color=color, italic=italic)
    cell.alignment = align(h=h, v=v, wrap=wrap)
    if bg:   cell.fill   = fill(bg)
    if border: cell.border = thin_border()

def hdr_row(ws, row, text, bg=NAVY, color=WHITE, size=13, cols=12):
    c = ws.cell(row=row, column=1, value=text)
    style(c, bold=True, size=size, color=color, bg=bg, h="left", v="center")
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=cols)
    ws.row_dimensions[row].height = 22

def label(ws, row, col, text, bold=False, bg=None, color="374151",
          wrap=False, size=11, italic=False, h="left"):
    c = ws.cell(row=row, column=col, value=text)
    style(c, bold=bold, size=size, color=color, bg=bg, wrap=wrap, italic=italic, h=h)
    return c

def inp(ws, row, col, value=None, fmt=None, bg=LYELLOW, border=True, h="right"):
    c = ws.cell(row=row, column=col, value=value)
    style(c, bold=True, size=11, color=NAVY, bg=bg, h=h, border=border)
    if fmt: c.number_format = fmt
    return c

def formula(ws, row, col, expr, fmt=None, bg=LGRAY, bold=False, color="374151"):
    c = ws.cell(row=row, column=col, value=expr)
    style(c, bold=bold, size=11, color=color, bg=bg, h="right")
    if fmt: c.number_format = fmt
    return c

wb = openpyxl.Workbook()

# ══════════════════════════════════════════════════════════════
# SHEET 1 — COVER
# ══════════════════════════════════════════════════════════════
cover = wb.active
cover.title = "Cover"
cover.sheet_view.showGridLines = False
cover.column_dimensions["A"].width = 32
cover.column_dimensions["B"].width = 38

hdr_row(cover, 1, "  BANK STOCK INTRINSIC VALUE CALCULATOR", bg=NAVY, size=15, cols=2)
hdr_row(cover, 2, "  Residual Income Model — Dividend-Irregularity Adjusted", bg=TEAL, size=12, cols=2)

data = [
    (4,  "Model",            "Residual Income Model (RIM)"),
    (5,  "Core Formula",     "IV = BV₀ + Σ[(ROE−r)×BVₜ₋₁/(1+r)ᵗ] + TV/(1+r)ⁿ"),
    (6,  "Cost of Equity",   "CAPM: r = Rf + β × (Rm − Rf)"),
    (7,  "ROE Forecast Modes","Manual | Auto from History | Growth Rate"),
    (8,  "Dividend Policy",  "No Dividends / Regular / Irregular (with irregularity factor)"),
    (10, "Intrinsic Value",  "=IF(Valuation!B14=0,\"Awaiting inputs\",Valuation!B14)"),
    (11, "Market Price",     "=IF(Inputs!B29=0,\"Not entered\",Inputs!B29)"),
    (12, "Margin of Safety", "=IFERROR(IF(Valuation!B14=0,\"—\",(Valuation!B14-Inputs!B29)/Valuation!B14),\"—\")"),
    (13, "Signal",           "=IFERROR(IF(Valuation!B14=0,\"—\",IF(Valuation!B14>Inputs!B29,\"UNDERVALUED\",IF(Valuation!B14<Inputs!B29,\"OVERVALUED\",\"FAIRLY VALUED\"))),\"—\")"),
]
for row, lbl, val in data:
    c1 = cover.cell(row=row, column=1, value=lbl)
    style(c1, bold=True, size=11, color=NAVY, bg=LGRAY2)
    c2 = cover.cell(row=row, column=2, value=val)
    style(c2, size=11, color="374151", bg=WHITE)

cover.cell(row=15, column=1, value="Note").font = font(italic=True, color="9CA3AF")
note = cover.cell(row=16, column=1, value="Enter data only on the Inputs sheet. Use the ROE_Engine sheet to set up your ROE forecast method.")
style(note, italic=True, size=10, color="9CA3AF", bg=WHITE, wrap=True)
cover.merge_cells("A16:B16")
cover.row_dimensions[16].height = 32

# ══════════════════════════════════════════════════════════════
# SHEET 2 — ROE_ENGINE  (the new comprehensive sheet)
# ══════════════════════════════════════════════════════════════
roe_eng = wb.create_sheet("ROE_Engine")
roe_eng.sheet_view.showGridLines = False
for col, w in [(1,30),(2,14),(3,14),(4,14),(5,14),(6,14),(7,20)]:
    roe_eng.column_dimensions[get_column_letter(col)].width = w

hdr_row(roe_eng, 1, "  ROE FORECAST ENGINE", bg=NAVY, size=14, cols=7)

# Mode selector
label(roe_eng, 3, 1, "ROE Forecast Mode", bold=True, size=11, color=NAVY)
mode_cell = inp(roe_eng, 3, 2, value="Manual", bg=LYELLOW)
dv = DataValidation(type="list", formula1='"Manual,Auto-Average,Auto-CAGR,Growth Rate"', allow_blank=False)
roe_eng.add_data_validation(dv)
dv.add(mode_cell)
label(roe_eng, 3, 3, "↑ Select forecast method", italic=True, size=9, color="9CA3AF")
roe_eng.merge_cells("C3:G3")

# ── Section A: Historical ROE ─────────────────────────────────
hdr_row(roe_eng, 5, "  SECTION A — Historical ROE (needed for Auto-Average and Auto-CAGR modes)", bg=TEAL, size=11, cols=7)
years_hist = ["Y-5 (oldest)","Y-4","Y-3","Y-2","Y-1 (latest)"]
for i, lbl in enumerate(years_hist):
    label(roe_eng, 6, i+2, lbl, bold=True, size=10, color=NAVY, bg=LGRAY2, h="center")
label(roe_eng, 7, 1, "Historical ROE (%)", bold=False, size=11)
for i in range(5):
    inp(roe_eng, 7, i+2, value=0, fmt="0.00%")

# Computed stats
label(roe_eng, 9, 1, "Simple Average of Historical ROEs →", italic=True, size=10, color=TEAL)
formula(roe_eng, 9, 2, "=IFERROR(AVERAGE(B7:F7),0)", fmt="0.00%", bold=True, color=TEAL)
label(roe_eng, 10, 1, "Latest ROE (Y-1) →", italic=True, size=10, color=TEAL)
formula(roe_eng, 10, 2, "=F7", fmt="0.00%", bold=True, color=TEAL)
label(roe_eng, 11, 1, "CAGR of ROE (Y-5 to Y-1) →", italic=True, size=10, color=TEAL)
formula(roe_eng, 11, 2, "=IFERROR(POWER(F7/B7,1/4)-1,0)", fmt="0.00%", bold=True, color=TEAL)
label(roe_eng, 11, 3, "(last/first)^(1/4)−1  — adjust denominator if fewer years used", italic=True, size=9, color="9CA3AF")
roe_eng.merge_cells("C11:G11")

# ── Section B: Growth Rate mode ───────────────────────────────
hdr_row(roe_eng, 13, "  SECTION B — Growth Rate Mode", bg=GOLD, color=NAVY, size=11, cols=7)
label(roe_eng, 14, 1, "Base ROE — latest known (%)", bold=False)
inp(roe_eng, 14, 2, value=0, fmt="0.00%")
label(roe_eng, 15, 1, "Annual ROE Growth Rate (%)")
inp(roe_eng, 15, 2, value=0, fmt="0.00%")

# ── Section C: Manual override ────────────────────────────────
hdr_row(roe_eng, 17, "  SECTION C — Manual ROE Forecast Override", bg=LGRAY2, color=NAVY, size=11, cols=7)
forecast_yrs = ["Year 1","Year 2","Year 3","Year 4","Year 5"]
for i, lbl in enumerate(forecast_yrs):
    label(roe_eng, 18, i+2, lbl, bold=True, size=10, color=NAVY, bg=LGRAY2, h="center")
label(roe_eng, 19, 1, "Manual ROE Forecast (%)", bold=False)
for i in range(5):
    inp(roe_eng, 19, i+2, value=0, fmt="0.00%")

# ── Section D: ACTIVE forecast (the one the model uses) ───────
hdr_row(roe_eng, 21, "  ACTIVE FORECAST — used by the model (auto-selected based on Mode)", bg=NAVY, size=11, cols=7)
label(roe_eng, 22, 1, "Selected ROE Forecast (%)", bold=True, color=NAVY)

# Formula: picks correct ROE based on mode
for i in range(5):
    col = i + 2
    col_let = get_column_letter(col)
    yr = i + 1
    # Manual: B19..F19; Average: B9; CAGR: F7*(1+B11)^t; Growth: B14*(1+B15)^t
    f = (f'=IF($B$3="Manual",{col_let}19,'
         f'IF($B$3="Auto-Average",$B$9,'
         f'IF($B$3="Auto-CAGR",IFERROR($F$7*POWER(1+$B$11,{yr}),0),'
         f'IF($B$3="Growth Rate",IFERROR($B$14*POWER(1+$B$15,{yr}),0),0))))')
    c = formula(roe_eng, 22, col, f, fmt="0.00%", bold=True, color=WHITE)
    c.fill = fill(NAVY)
    c.font = Font(bold=True, size=11, color=WHITE, name="Calibri")

label(roe_eng, 23, 1, "↑ These values feed directly into the Inputs sheet ROE rows.", italic=True, size=9, color="9CA3AF")
roe_eng.merge_cells("A23:G23")

# ══════════════════════════════════════════════════════════════
# SHEET 3 — INPUTS
# ══════════════════════════════════════════════════════════════
inp_ws = wb.create_sheet("Inputs")
inp_ws.sheet_view.showGridLines = False
for col, w in [(1,38),(2,18),(3,18),(4,18),(5,18),(6,18)]:
    inp_ws.column_dimensions[get_column_letter(col)].width = w

hdr_row(inp_ws, 1, "  INPUTS & ASSUMPTIONS", bg=NAVY, size=14, cols=6)

# Setup
hdr_row(inp_ws, 3, "  1. SETUP", bg=TEAL, size=11, cols=6)
rows_setup = [
    (4,  "Currency",                         "GHS",    None,   False),
    (5,  "Number of forecast years (3–5)",   5,        "0",    True),
]
for row, lbl, val, fmt_, editable in rows_setup:
    label(inp_ws, row, 1, lbl)
    c = inp_ws.cell(row=row, column=2, value=val)
    style(c, bold=True, size=11, color=NAVY, bg=LYELLOW if editable else LGRAY, h="right")
    if fmt_: c.number_format = fmt_

dv_years = DataValidation(type="whole", operator="between", formula1="3", formula2="5")
inp_ws.add_data_validation(dv_years)
dv_years.add(inp_ws["B5"])

# CAPM
hdr_row(inp_ws, 7, "  2. COST OF EQUITY — CAPM  [r = Rf + β × (Rm − Rf)]", bg=TEAL, size=11, cols=6)
capm_rows = [
    (8,  "Risk-free rate — long-term govt bond yield (%)", 0,   "0.00%", True),
    (9,  "Expected market return (%)",                     0,   "0.00%", True),
    (10, "Beta (stock volatility vs market)",              0,   "0.00",  True),
    (11, "Market risk premium (MRP = Rm − Rf)",           "=B9-B8", "0.00%", False),
    (12, "Cost of Equity, r = Rf + β × MRP",              "=B8+B10*(B9-B8)", "0.00%", False),
]
for row, lbl, val, fmt_, editable in capm_rows:
    label(inp_ws, row, 1, lbl, bold=not editable)
    c = inp_ws.cell(row=row, column=2, value=val)
    style(c, bold=True, size=11, color=NAVY if editable else TEAL,
          bg=LYELLOW if editable else LGRAY, h="right")
    c.number_format = fmt_

# RIM Inputs
hdr_row(inp_ws, 14, "  3. RESIDUAL INCOME INPUTS", bg=TEAL, size=11, cols=6)
rim_rows = [
    (15, "Current Book Value per Share (BV₀)",    0,   "0.00",  True),
    (16, "Dividend Policy",                       "Regular Dividends", None, True),
    (17, "Dividend Payout Ratio — stated (%)",    0,   "0.00%", True),
    (18, "Irregularity Factor (0–1; for Irregular policy)", 1, "0.00", True),
    (19, "Effective Payout used in model",        "=IF(B16=\"No Dividends\",0,IF(B16=\"Irregular Dividends\",B17*B18,B17))", "0.00%", False),
]
for row, lbl, val, fmt_, editable in rim_rows:
    label(inp_ws, row, 1, lbl, bold=not editable)
    c = inp_ws.cell(row=row, column=2, value=val)
    style(c, bold=True, size=11, color=NAVY if editable else TEAL,
          bg=LYELLOW if editable else LGRAY, h="right")
    if fmt_: c.number_format = fmt_

dv_pol = DataValidation(type="list", formula1='"No Dividends,Regular Dividends,Irregular Dividends"')
inp_ws.add_data_validation(dv_pol)
dv_pol.add(inp_ws["B16"])
label(inp_ws, 18, 3, "1.0 = dividends paid every year; 0.75 = 3 out of 4 years", italic=True, size=9, color="9CA3AF")
inp_ws.merge_cells("C18:F18")

# Terminal Value
hdr_row(inp_ws, 21, "  4. TERMINAL VALUE ASSUMPTIONS", bg=TEAL, size=11, cols=6)
tv_rows = [
    (22, "Terminal growth rate, g (%)",          0,   "0.00%", True),
    (23, "Long-term sustainable ROE (%)",         0,   "0.00%", True),
    (24, "CHECK: LT ROE > Cost of Equity?",      '=IF(B23>B12,"✅ YES — value-creating","❌ NO — destroys value")', None, False),
    (25, "CHECK: g < Cost of Equity?",           '=IF(B22<B12,"✅ YES — model valid","❌ NO — fix g or CAPM inputs")', None, False),
]
for row, lbl, val, fmt_, editable in tv_rows:
    label(inp_ws, row, 1, lbl, bold=not editable)
    c = inp_ws.cell(row=row, column=2, value=val)
    style(c, bold=True, size=11, color=NAVY if editable else TEAL,
          bg=LYELLOW if editable else LGRAY, h="right" if editable else "left")
    if fmt_: c.number_format = fmt_

# Market Price
hdr_row(inp_ws, 27, "  5. MARKET COMPARISON (optional)", bg=TEAL, size=11, cols=6)
label(inp_ws, 28, 1, "Current market price per share")
inp(inp_ws, 28, 2, value=0, fmt="0.00")

# ROE Forecast — links to ROE_Engine
hdr_row(inp_ws, 30, "  6. ROE FORECAST BY YEAR — linked to ROE_Engine (change mode there)", bg=GOLD, color=NAVY, size=11, cols=6)
yr_labels = ["Year 1","Year 2","Year 3","Year 4","Year 5"]
for i, lbl in enumerate(yr_labels):
    label(inp_ws, 31, i+2, lbl, bold=True, size=10, color=NAVY, bg=LGRAY2, h="center")
label(inp_ws, 32, 1, "Active Forecasted ROE", bold=True, color=NAVY)
for i in range(5):
    col_let = get_column_letter(i+2)
    eng_col = get_column_letter(i+2)
    c = formula(inp_ws, 32, i+2, f"=ROE_Engine!{eng_col}22", fmt="0.00%", bold=True, color=NAVY)
    c.fill = fill(LBLUE)

label(inp_ws, 33, 1, "Override: Manual ROE entry here (type directly to override ROE_Engine link)", italic=True, size=9, color="9CA3AF")
inp_ws.merge_cells("A33:F33")

# ══════════════════════════════════════════════════════════════
# SHEET 4 — FORECAST
# ══════════════════════════════════════════════════════════════
fc = wb.create_sheet("Forecast")
fc.sheet_view.showGridLines = False
for col, w in [(1,38),(2,16),(3,16),(4,16),(5,16),(6,16)]:
    fc.column_dimensions[get_column_letter(col)].width = w

hdr_row(fc, 1, "  FORECAST ENGINE — Book Value Roll-Forward & Residual Income", bg=NAVY, size=13, cols=6)
label(fc, 2, 1, "All inputs drawn from Inputs sheet. Years beyond forecast horizon are hidden.", italic=True, size=9, color="9CA3AF")

for i in range(5):
    col = i+2
    c = fc.cell(row=3, column=col, value=f"Year {i+1}")
    style(c, bold=True, size=10, color=WHITE, bg=NAVY, h="center")

fc_rows = [
    (4,  "Active in forecast?",             [f'=IF({i+1}<=Inputs!$B$5,"Yes","—")' for i in range(5)],     None,    False),
    (5,  "Forecasted ROE (%)",              [f"=IF({i+1}<=Inputs!$B$5,Inputs!{get_column_letter(i+2)}32,\"\")" for i in range(5)], "0.00%", False),
    (6,  "Opening Book Value per share",    None, "0.00", False),
    (7,  "Net Income per share (ROE×BV)",   None, "0.00", False),
    (8,  "Dividend per share (eff.payout)", None, "0.00", False),
    (9,  "Retained Earnings per share",     None, "0.00", False),
    (10, "Closing Book Value per share",    None, "0.00", False),
    (11, "Equity Charge (r × opening BV)", None, "0.00", False),
    (12, "Residual Income per share",       None, "0.00", False),
    (13, "Discount factor  1/(1+r)ᵗ",      None, "0.0000", False),
    (14, "PV of Residual Income",           None, "0.00", False),
]

label_bgs = {4:LGRAY2, 5:LBLUE, 6:LGRAY, 7:LGRAY, 8:LGRAY, 9:LGRAY, 10:LGRAY2, 11:LGRAY, 12:LGREEN, 13:LGRAY, 14:LGREEN}
for row, lbl, formulas_, fmt_, _ in fc_rows:
    label(fc, row, 1, lbl, bold=(row in [6,10,12,14]), size=10)
    for i in range(5):
        col = i+2
        if formulas_:
            f = formulas_[i]
        else:
            col_let = get_column_letter(col)
            prev_col = get_column_letter(col-1)
            r = row
            if r == 6:   # Opening BV
                f = (f'=IF({i+1}>Inputs!$B$5,"",IF({i+1}=1,Inputs!$B$15,{prev_col}10))' if i > 0
                     else '=IF(1<=Inputs!$B$5,Inputs!$B$15,"")')
            elif r == 7: f = f'=IF({col_let}4="—","",{col_let}5*{col_let}6)'
            elif r == 8: f = f'=IF({col_let}4="—","",{col_let}7*Inputs!$B$19)'
            elif r == 9: f = f'=IF({col_let}4="—","",{col_let}7-{col_let}8)'
            elif r == 10:f = f'=IF({col_let}4="—","",{col_let}6+{col_let}9)'
            elif r == 11:f = f'=IF({col_let}4="—","",Inputs!$B$12*{col_let}6)'
            elif r == 12:f = f'=IF({col_let}4="—","",{col_let}7-{col_let}11)'
            elif r == 13:f = f'=IF({col_let}4="—","",1/POWER(1+Inputs!$B$12,{i+1}))'
            elif r == 14:f = f'=IF({col_let}4="—","",{col_let}12*{col_let}13)'
            else: f = ""

        c = fc.cell(row=row, column=col, value=f)
        bg = label_bgs.get(row, LGRAY)
        style(c, size=10, color="374151", bg=bg, h="right")
        if fmt_: c.number_format = fmt_

# ══════════════════════════════════════════════════════════════
# SHEET 5 — VALUATION
# ══════════════════════════════════════════════════════════════
val = wb.create_sheet("Valuation")
val.sheet_view.showGridLines = False
for col, w in [(1,40),(2,20),(3,20)]:
    val.column_dimensions[get_column_letter(col)].width = w

hdr_row(val, 1, "  VALUATION SUMMARY", bg=NAVY, size=14, cols=3)
label(val, 2, 1, "IV = BV₀ + Σ PV(Residual Income) + PV(Terminal Value)", bold=False, italic=True, size=10, color="9CA3AF")

label(val, 4, 1, "Cost of Equity, r",              bold=True, color=NAVY)
formula(val, 4, 2, "=Inputs!B12", fmt="0.00%", bold=True, color=TEAL)
label(val, 5, 1, "Forecast horizon (years)",        bold=True, color=NAVY)
formula(val, 5, 2, "=Inputs!B5", fmt="0", bold=True, color=TEAL)

hdr_row(val, 7, "  Build-up of Intrinsic Value", bg=TEAL, size=11, cols=3)
bup = [
    (8,  "Current Book Value per share (BV₀)",        "=Inputs!B15",  "0.00"),
    (9,  "PV of Residual Income — forecast period",   "=IFERROR(SUM(Forecast!B14:F14),0)", "0.00"),
    (10, "Terminal BV (end of final forecast year)",  "=IFERROR(CHOOSE(Inputs!B5,Forecast!B10,Forecast!C10,Forecast!D10,Forecast!E10,Forecast!F10),0)", "0.00"),
    (11, "Terminal Residual Income (first perp. year)","=IFERROR((Inputs!B23-Inputs!B12)*B10,0)", "0.00"),
    (12, "Terminal Value = RI/(r−g)",                 "=IFERROR(B11*(1+Inputs!B22)/(Inputs!B12-Inputs!B22),0)", "0.00"),
    (13, "PV of Terminal Value",                      "=IFERROR(B12/POWER(1+Inputs!B12,Inputs!B5),0)", "0.00"),
    (14, "INTRINSIC VALUE PER SHARE",                 "=IFERROR(B8+B9+B13,0)", "0.00"),
]
for row, lbl, fml, fmt_ in bup:
    is_iv = (row == 14)
    label(val, row, 1, lbl, bold=is_iv, size=11 if is_iv else 10,
          color=NAVY if is_iv else "374151", bg=LBLUE if is_iv else None)
    c = val.cell(row=row, column=2, value=fml)
    style(c, bold=is_iv, size=13 if is_iv else 11, color=TEAL,
          bg=LBLUE if is_iv else LGRAY, h="right")
    c.number_format = fmt_
    if is_iv:
        val.row_dimensions[row].height = 24

hdr_row(val, 16, "  Market Comparison", bg=TEAL, size=11, cols=3)
comp = [
    (17, "Current market price per share",          "=Inputs!B28",                                          "0.00"),
    (18, "Upside / (downside) per share",           "=IFERROR(B14-B17,\"—\")",                              "0.00"),
    (19, "Margin of Safety (% of intrinsic value)", "=IFERROR((B14-B17)/B14,\"—\")",                        "0.00%"),
    (20, "Price / Book Value",                      "=IFERROR(B17/Inputs!B15,\"—\")",                       "0.00x"),
    (21, "Intrinsic / Book Value",                  "=IFERROR(B14/Inputs!B15,\"—\")",                       "0.00x"),
    (22, "Signal",                                  '=IFERROR(IF(B14=0,"—",IF(B14>B17,"UNDERVALUED","OVERVALUED")),"—")', None),
]
for row, lbl, fml, fmt_ in comp:
    label(val, row, 1, lbl, size=10)
    c = val.cell(row=row, column=2, value=fml)
    style(c, bold=(row==22), size=11, color=TEAL, bg=LGRAY, h="right")
    if fmt_: c.number_format = fmt_

# ══════════════════════════════════════════════════════════════
# SHEET 6 — SENSITIVITY
# ══════════════════════════════════════════════════════════════
sens = wb.create_sheet("Sensitivity")
sens.sheet_view.showGridLines = False
for col, w in [(1,4),(2,22),(3,14),(4,14),(5,14),(6,14),(7,14),(8,14)]:
    sens.column_dimensions[get_column_letter(col)].width = w

hdr_row(sens, 1, "  SENSITIVITY ANALYSIS — Intrinsic Value per Share", bg=NAVY, size=13, cols=8)

# Table A: Terminal Growth (rows) × LT ROE (cols)
label(sens, 3, 2, "TABLE A  —  Terminal Growth Rate (rows)  ×  Long-Term ROE (cols)", bold=True, color=NAVY, size=11)
sens.merge_cells("B3:H3")

lt_roe_offsets = [-0.04, -0.02, 0, 0.02, 0.04]
g_offsets      = [-0.02, -0.01, 0, 0.01, 0.02]

label(sens, 4, 2, "g \\ LT ROE →", bold=True, bg=LGRAY2, size=10, h="center")
for j, d in enumerate(lt_roe_offsets):
    c = sens.cell(row=4, column=3+j, value=f"=TEXT(Inputs!B23+{d},\"0.0%\")")
    style(c, bold=True, size=10, color=WHITE, bg=NAVY, h="center")

for i, gd in enumerate(g_offsets):
    row = 5+i
    gc = sens.cell(row=row, column=2, value=f"=TEXT(Inputs!B22+{gd},\"0.0%\")")
    style(gc, bold=True, size=10, color=NAVY, bg=LGRAY2, h="center")
    for j, rd in enumerate(lt_roe_offsets):
        g_expr   = f"(Inputs!B22+{gd})"
        roe_expr = f"(Inputs!B23+{rd})"
        r_expr   = "Inputs!B12"
        # IV = BV0 + PV_RI + PV_TV; approximate: use Valuation sheet RI and recalc TV only
        # Full formula: BV0 + SUM(PV_RI_forecast) + ((LT_ROE-r)*terminal_BV*(1+g))/(r-g)/(1+r)^n
        f = (f"=IFERROR(IF({g_expr}>={r_expr},\"N/A\","
             f"IF({roe_expr}<={r_expr},\"N/A\","
             f"Inputs!B15+SUM(Forecast!B14:F14)+"
             f"(({roe_expr}-{r_expr})*Valuation!B10*(1+{g_expr}))/({r_expr}-{g_expr})"
             f"/POWER(1+{r_expr},Inputs!B5))),\"N/A\")")
        c = sens.cell(row=row, column=3+j, value=f)
        style(c, size=10, color="374151", bg=WHITE, h="right", border=True)
        c.number_format = "0.00"

# Table B: Beta (rows) × Terminal Growth (cols)
label(sens, 12, 2, "TABLE B  —  Beta (rows)  ×  Terminal Growth Rate (cols)", bold=True, color=NAVY, size=11)
sens.merge_cells("B12:H12")

beta_offsets = [-0.4,-0.2,0,0.2,0.4]
label(sens, 13, 2, "β \\ g →", bold=True, bg=LGRAY2, size=10, h="center")
for j, d in enumerate(g_offsets):
    c = sens.cell(row=13, column=3+j, value=f"=TEXT(Inputs!B22+{d},\"0.0%\")")
    style(c, bold=True, size=10, color=WHITE, bg=GOLD, h="center")
    c.font = Font(bold=True, size=10, color=NAVY, name="Calibri")

for i, bd in enumerate(beta_offsets):
    row = 14+i
    beta_val = f"MAX(0.01,Inputs!B10+{bd})"
    r_new    = f"(Inputs!B8+({beta_val})*(Inputs!B9-Inputs!B8))"
    brow = sens.cell(row=row, column=2, value=f"=TEXT(MAX(0.01,Inputs!B10+{bd}),\"0.00\")")
    style(brow, bold=True, size=10, color=NAVY, bg=LGRAY2, h="center")
    for j, gd in enumerate(g_offsets):
        g_expr = f"(Inputs!B22+{gd})"
        f = (f"=IFERROR(IF({g_expr}>={r_new},\"N/A\","
             f"IF(Inputs!B23<={r_new},\"N/A\","
             f"Inputs!B15+SUM(Forecast!B14:F14)+"
             f"((Inputs!B23-{r_new})*Valuation!B10*(1+{g_expr}))/({r_new}-{g_expr})"
             f"/POWER(1+{r_new},Inputs!B5))),\"N/A\")")
        c = sens.cell(row=row, column=3+j, value=f)
        style(c, size=10, color="374151", bg=WHITE, h="right", border=True)
        c.number_format = "0.00"

# ══════════════════════════════════════════════════════════════
# SHEET 7 — INSTRUCTIONS
# ══════════════════════════════════════════════════════════════
instr = wb.create_sheet("Instructions")
instr.sheet_view.showGridLines = False
instr.column_dimensions["A"].width = 10
instr.column_dimensions["B"].width = 70

hdr_row(instr, 1, "  HOW TO USE THIS MODEL", bg=NAVY, size=14, cols=2)

steps = [
    ("Step 1", "Select ROE Forecast Mode (ROE_Engine sheet)",
     "Open the ROE_Engine sheet and select your preferred mode from the dropdown in cell B3:\n"
     "• Manual — type each year's ROE directly into the Inputs sheet (row 32)\n"
     "• Auto-Average — enter 3–5 historical ROEs; model uses their simple average\n"
     "• Auto-CAGR — enter historical ROEs; model projects using the CAGR trend\n"
     "• Growth Rate — enter a base ROE and annual growth rate; model compounds forward"),
    ("Step 2", "Enter CAPM Inputs (Inputs sheet)",
     "Risk-free rate: yield on 5–10 year government bonds (e.g. Ghana 10yr bond)\n"
     "Market return: expected annual return of the stock market index\n"
     "Beta: stock's volatility relative to the market (from Bloomberg, Reuters, or Yahoo Finance)\n"
     "Cost of Equity is calculated automatically."),
    ("Step 3", "Enter Residual Income Inputs (Inputs sheet)",
     "Book Value per Share: Total Equity ÷ Shares Outstanding (from the latest Balance Sheet)\n"
     "Dividend Policy: No Dividends / Regular / Irregular\n"
     "Payout Ratio: % of EPS distributed as dividends\n"
     "Irregularity Factor: fraction of years dividends are actually paid (Irregular policy only)\n"
     "Effective Payout = Stated Payout × Irregularity Factor (calculated automatically)"),
    ("Step 4", "Enter Terminal Value Assumptions (Inputs sheet)",
     "Terminal Growth Rate: the rate at which residual income grows beyond the forecast horizon.\n"
     "  → Must be below the Cost of Equity and below long-run GDP growth\n"
     "Long-Term ROE: the sustainable ROE the bank achieves in perpetuity.\n"
     "  → Must exceed Cost of Equity; otherwise the company destroys value in perpetuity"),
    ("Step 5", "Read Results (Valuation & Sensitivity sheets)",
     "Valuation sheet: shows the intrinsic value build-up (BV₀ + PV(RI) + PV(TV))\n"
     "Sensitivity Table A: intrinsic value across terminal growth × long-term ROE combinations\n"
     "Sensitivity Table B: intrinsic value across beta × terminal growth combinations\n"
     "Cover sheet: shows headline intrinsic value, market price, margin of safety, and signal"),
    ("Convention", "Input cells only",
     "Yellow cells = data entry (your inputs)\n"
     "All other cells contain formulas — do not overwrite them\n"
     "Enter all percentages as decimals (e.g. 18% → 0.18) or as whole numbers (18) depending on the cell format shown"),
]

row = 3
for step, title, detail in steps:
    c1 = instr.cell(row=row, column=1, value=step)
    style(c1, bold=True, size=10, color=WHITE, bg=TEAL, h="center", v="top")
    c2 = instr.cell(row=row, column=2, value=title)
    style(c2, bold=True, size=11, color=NAVY, bg=LBLUE)
    row += 1
    c3 = instr.cell(row=row, column=2, value=detail)
    style(c3, size=10, color="374151", wrap=True, v="top")
    instr.row_dimensions[row].height = max(60, detail.count('\n')*16+20)
    row += 2

# Reorder sheets
wb._sheets = [cover, instr, wb["ROE_Engine"], inp_ws, fc, val, sens]

wb.save(OUT)
print(f"Saved: {OUT}")

#!/usr/bin/env python3
"""Check what the normal text margins are on a typical text-only page."""
import fitz
doc = fitz.open("tex files/IIT_Madras_MTech_in_Industrial_AI_Thesis_Template/thesis.pdf")

# Check a few normal text pages to establish baseline margins
for pn in [20, 30, 40, 50, 91, 93]:
    page = doc[pn - 1]
    blocks = page.get_text("dict")["blocks"]
    min_x0 = float('inf')
    max_x1 = float('-inf')
    for block in blocks:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    x0 = span["bbox"][0]
                    x1 = span["bbox"][2]
                    if x0 < min_x0: min_x0 = x0
                    if x1 > max_x1: max_x1 = x1
    print(f"Page {pn}: x0={min_x0:.2f}  x1={max_x1:.2f}  width={max_x1-min_x0:.2f}")

doc.close()

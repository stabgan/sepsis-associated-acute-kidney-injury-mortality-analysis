#!/usr/bin/env python3
"""Check all bibliography pages for left margin issues."""
import fitz
doc = fitz.open("tex files/IIT_Madras_MTech_in_Industrial_AI_Thesis_Template/thesis.pdf")
total = doc.page_count
print(f"Total pages: {total}")

for pn in range(max(1, total - 15), total + 1):
    page = doc[pn - 1]
    blocks = page.get_text("dict")["blocks"]
    min_x0 = float('inf')
    max_x1 = float('-inf')
    min_text = ""
    max_text = ""
    for block in blocks:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    x0 = span["bbox"][0]
                    x1 = span["bbox"][2]
                    if x0 < min_x0:
                        min_x0 = x0
                        min_text = span["text"][:50]
                    if x1 > max_x1:
                        max_x1 = x1
                        max_text = span["text"][:50]
        elif block["type"] == 1:
            x0 = block["bbox"][0]
            x1 = block["bbox"][2]
            if x1 > max_x1:
                max_x1 = x1
                max_text = f"[IMAGE w={x1-x0:.1f}]"
    flag = ""
    if min_x0 < 72: flag += " *** LEFT MARGIN"
    if max_x1 > 524: flag += " *** RIGHT MARGIN"
    if flag:
        print(f"Page {pn}: x0={min_x0:.2f} x1={max_x1:.2f} -> left='{min_text}' right='{max_text}'{flag}")

doc.close()

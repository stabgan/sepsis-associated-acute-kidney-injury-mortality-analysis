#!/usr/bin/env python3
"""Check ALL pages for margin violations."""
import fitz
doc = fitz.open("tex files/IIT_Madras_MTech_in_Industrial_AI_Thesis_Template/thesis.pdf")
total = doc.page_count
print(f"Total pages: {total}")

LEFT_LIMIT = 63.0   # minimum allowed x0
RIGHT_LIMIT = 524.0  # maximum allowed x1

violations = []
for pn in range(1, total + 1):
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
                        min_text = span["text"][:60]
                    if x1 > max_x1:
                        max_x1 = x1
                        max_text = span["text"][:60]
        elif block["type"] == 1:
            x0 = block["bbox"][0]
            x1 = block["bbox"][2]
            if x0 < min_x0:
                min_x0 = x0
                min_text = f"[IMAGE w={x1-x0:.1f}]"
            if x1 > max_x1:
                max_x1 = x1
                max_text = f"[IMAGE w={x1-x0:.1f}]"
    
    flags = []
    if min_x0 < LEFT_LIMIT: flags.append(f"LEFT x0={min_x0:.2f}")
    if max_x1 > RIGHT_LIMIT: flags.append(f"RIGHT x1={max_x1:.2f}")
    if flags:
        violations.append(f"Page {pn}: {', '.join(flags)} -> '{min_text}' / '{max_text}'")

if violations:
    print(f"\n{len(violations)} pages with margin violations:")
    for v in violations:
        print(f"  {v}")
else:
    print("\nNo margin violations found!")

doc.close()

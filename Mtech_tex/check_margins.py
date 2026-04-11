#!/usr/bin/env python3
"""Analyze PDF pages for margin violations by measuring actual content bounds."""
import fitz  # PyMuPDF
import sys

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "tex files/IIT_Madras_MTech_in_Industrial_AI_Thesis_Template/thesis.pdf"
doc = fitz.open(pdf_path)

# Check specific pages (0-indexed internally, 1-indexed in report)
pages_to_check = [92, 94, 96, 97, 142]

for page_num in pages_to_check:
    page = doc[page_num - 1]  # 0-indexed
    rect = page.rect  # full page rectangle
    
    # Get all text blocks with positions
    blocks = page.get_text("dict")["blocks"]
    
    min_x0 = float('inf')
    max_x1 = float('-inf')
    min_x0_text = ""
    max_x1_text = ""
    
    for block in blocks:
        if block["type"] == 0:  # text block
            for line in block["lines"]:
                for span in line["spans"]:
                    x0 = span["bbox"][0]
                    x1 = span["bbox"][2]
                    text = span["text"][:80]
                    if x0 < min_x0:
                        min_x0 = x0
                        min_x0_text = text
                    if x1 > max_x1:
                        max_x1 = x1
                        max_x1_text = text
        elif block["type"] == 1:  # image block
            x0 = block["bbox"][0]
            x1 = block["bbox"][2]
            if x0 < min_x0:
                min_x0 = x0
                min_x0_text = f"[IMAGE at ({x0:.1f}, {block['bbox'][1]:.1f})]"
            if x1 > max_x1:
                max_x1 = x1
                max_x1_text = f"[IMAGE at ({x0:.1f}, {block['bbox'][1]:.1f}), w={x1-x0:.1f}]"
    
    print(f"\n=== Page {page_num} ===")
    print(f"  Page size: {rect.width:.1f} x {rect.height:.1f}")
    print(f"  Content x-range: {min_x0:.2f} to {max_x1:.2f}")
    print(f"  Leftmost content:  x0={min_x0:.2f} -> '{min_x0_text}'")
    print(f"  Rightmost content: x1={max_x1:.2f} -> '{max_x1_text}'")

doc.close()

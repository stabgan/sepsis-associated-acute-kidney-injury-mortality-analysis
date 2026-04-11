#!/usr/bin/env python3
"""Identify images on problem pages."""
import fitz
doc = fitz.open("tex files/IIT_Madras_MTech_in_Industrial_AI_Thesis_Template/thesis.pdf")

for pn in [92, 94, 96, 97]:
    page = doc[pn - 1]
    images = page.get_images(full=True)
    blocks = page.get_text("dict")["blocks"]
    print(f"\n=== Page {pn} ===")
    for block in blocks:
        if block["type"] == 1:
            bbox = block["bbox"]
            print(f"  Image bbox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}) w={bbox[2]-bbox[0]:.1f}")
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    t = span["text"].strip()
                    if t and ("Figure" in t or "Table" in t or "plot" in t.lower() or "image" in t.lower()):
                        print(f"  Caption text: '{t[:100]}'")
                        break

doc.close()

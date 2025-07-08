import sys
import os
from pathlib import Path
from docx import Document
from app import convert_pdf_to_docx

import tkinter as tk
from tkinter import filedialog, messagebox

def docx_to_txt(docx_path, txt_path=None):
    """
    Convert a DOCX file to a plain TXT file.
    """
    doc = Document(docx_path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)
    txt_content = "\n".join(lines)
    if not txt_path:
        txt_path = Path(docx_path).with_suffix('.txt')
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    return str(txt_path)

def run_gui():
    def select_pdf():
        file_path = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf")]
        )
        if file_path:
            pdf_var.set(file_path)

    def select_output_docx():
        file_path = filedialog.asksaveasfilename(
            title="Save DOCX as...",
            defaultextension=".docx",
            filetypes=[("DOCX files", "*.docx")]
        )
        if file_path:
            docx_var.set(file_path)

    def select_output_txt():
        file_path = filedialog.asksaveasfilename(
            title="Save TXT as...",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            txt_var.set(file_path)

    def convert():
        pdf_path = pdf_var.get()
        docx_path = docx_var.get() or None
        txt_path = txt_var.get() or None

        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file.")
            return

        try:
            # Step 1: PDF to DOCX
            docx_result = convert_pdf_to_docx(pdf_path=pdf_path, output_path=docx_path)
            # Step 2: DOCX to TXT
            txt_result = docx_to_txt(docx_result, txt_path)
            messagebox.showinfo("Success", f"TXT created:\n{txt_result}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("PDF → DOCX → TXT Converter")

    pdf_var = tk.StringVar()
    docx_var = tk.StringVar()
    txt_var = tk.StringVar()

    tk.Label(root, text="PDF file:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=pdf_var, width=40).grid(row=0, column=1)
    tk.Button(root, text="Browse...", command=select_pdf).grid(row=0, column=2)

    tk.Label(root, text="DOCX output (optional):").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=docx_var, width=40).grid(row=1, column=1)
    tk.Button(root, text="Browse...", command=select_output_docx).grid(row=1, column=2)

    tk.Label(root, text="TXT output (optional):").grid(row=2, column=0, sticky="e")
    tk.Entry(root, textvariable=txt_var, width=40).grid(row=2, column=1)
    tk.Button(root, text="Browse...", command=select_output_txt).grid(row=2, column=2)

    tk.Button(root, text="Convert", command=convert, width=20).grid(row=3, column=0, columnspan=3, pady=10)

    root.mainloop()

if __name__ == "__main__":
    # Run the GUI
    run_gui()
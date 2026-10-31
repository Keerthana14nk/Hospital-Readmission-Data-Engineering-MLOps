"""
Script to compile the 32-page report HTML and render the PDF via Edge Headless.
"""
import os
import sys
import base64
import subprocess
from pathlib import Path
import fitz

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "report_assets"

def get_base64_image(file_path):
    p = Path(file_path)
    if not p.exists():
        print(f"Warning: image file not found: {p}")
        return ""
    suffix = p.suffix.lower().replace('.', '')
    if suffix == 'jpg':
        suffix = 'jpeg'
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/{suffix};base64,{data}"

def main():
    template_file = BASE_DIR / "report_template.html"
    if not template_file.exists():
        print(f"Template not found: {template_file}")
        return

    print("Reading template...")
    html_content = template_file.read_text(encoding="utf-8")

    replacements = {
        "{{VTU_LOGO}}": get_base64_image(ASSETS_DIR / "vtu_logo.jpg"),
        "{{CMRIT_LOGO}}": get_base64_image(ASSETS_DIR / "cmrit_logo.png"),
        "{{FIG1}}": get_base64_image(ASSETS_DIR / "fig1_architecture.png"),
        "{{FIG2}}": get_base64_image(ASSETS_DIR / "fig2_schema.png"),
        "{{FIG3}}": get_base64_image(ASSETS_DIR / "fig3_dashboard.png"),
        "{{FIG4}}": get_base64_image(ASSETS_DIR / "fig4_mlflow.png"),
        "{{FIG5}}": get_base64_image(ASSETS_DIR / "fig5_fastapi.png"),
        "{{FIG6}}": get_base64_image(ASSETS_DIR / "fig6_airflow.png"),
    }

    print("Replacing placeholders...")
    for placeholder, b64 in replacements.items():
        html_content = html_content.replace(placeholder, b64)

    output_html = BASE_DIR / "Hospital_Readmission_Project_Report.html"
    output_pdf = BASE_DIR / "Hospital_Readmission_Project_Report.pdf"

    output_html.write_text(html_content, encoding="utf-8")
    print(f"Saved compiled HTML to: {output_html}")

    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not Path(edge_exe).exists():
        print(f"Edge executable not found at: {edge_exe}")
        return

    print("Rendering PDF with Edge Headless...")
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf.resolve()}",
        output_html.resolve().as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Edge error: {res.stderr}")
        return

    if not output_pdf.exists():
        print("PDF was not created!")
        return

    file_size_mb = output_pdf.stat().st_size / (1024 * 1024)
    print(f"PDF Successfully Generated: {output_pdf} ({file_size_mb:.2f} MB)")

    # Verify PDF with PyMuPDF
    doc = fitz.open(str(output_pdf))
    print(f"Total Page Count: {len(doc)}")
    for i in [0, 1, 2, 6, 7, 11, 20, 25, 30, len(doc)-1]:
        if i < len(doc):
            first_line = doc[i].get_text().splitlines()
            header_snippet = first_line[0] if first_line else "EMPTY"
            print(f"  Page {i+1} Header: {header_snippet[:60]}")

if __name__ == "__main__":
    main()

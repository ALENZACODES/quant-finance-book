"""Strip all metadata from PDFs and PNGs, then set Author = John Khan and nothing else.

PDF: the XMP stream and every Info key are removed, then /Author is the only Info entry; the file is
rewritten (not incrementally updated) with a deterministic ID so no timestamp or tool name survives.
PNG: all ancillary text and time chunks are dropped by exiftool, then a single Author tag is written.
Usage: python scripts/strip_meta.py file1.pdf file2.png ...
"""
import subprocess, sys
from pathlib import Path
import pikepdf

AUTHOR = "John Khan"


def clean_pdf(path: Path):
    tmp = path.with_suffix(".tmp.pdf")
    with pikepdf.open(path) as pdf:
        if "/Metadata" in pdf.Root:
            del pdf.Root.Metadata
        for page in pdf.pages:                       # per-page XMP or PieceInfo, if any
            for key in ("/Metadata", "/PieceInfo"):
                if key in page.obj:
                    del page.obj[key]
        pdf.docinfo = pdf.make_indirect(pikepdf.Dictionary(Author=pikepdf.String(AUTHOR)))
        pdf.save(tmp, static_id=True, fix_metadata_version=False, linearize=False,
                 object_stream_mode=pikepdf.ObjectStreamMode.generate)
    tmp.replace(path)


def clean_png(path: Path):
    subprocess.run(["exiftool", "-q", "-overwrite_original", "-all=", str(path)], check=True)
    subprocess.run(["exiftool", "-q", "-overwrite_original", f"-PNG:Author={AUTHOR}", str(path)], check=True)


if __name__ == "__main__":
    for f in map(Path, sys.argv[1:]):
        if f.suffix.lower() == ".pdf":
            clean_pdf(f)
        elif f.suffix.lower() == ".png":
            clean_png(f)

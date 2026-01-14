import cairosvg
from PIL import Image
import io
import os
from pathlib import Path

def convert_single(path: str, w, h):
    png_bytes = cairosvg.svg2png(url=path, output_width=w, output_height=h)

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    bmp_path = path.replace(".svg", f".bmp").replace("full-deck", "full-deck-bmp")
    img.save(bmp_path)

def main():
    w, h = 130, 182

    for p in Path("assets/full-deck/").glob('*'):
        print(p)
        convert_single(p.as_posix(), w, h)

if __name__ == '__main__':
    main()
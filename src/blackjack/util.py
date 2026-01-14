import sys
import io
import pygame
import functools
import cairosvg
from .constants import *

@functools.cache
def load_svg(path, size=(130, CARD_HEIGHT)):
    w, h = size
    png_bytes = cairosvg.svg2png(url=path, output_width=w, output_height=h)
    return pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()

@functools.cache
def load_bmp(path):
    return pygame.image.load(path)

@functools.cache
def load_png(path):
    return pygame.image.load(path).convert_alpha()

@functools.cache
def get_font(size=48):
    return pygame.font.Font(None, size)

def render_text(s, color, **posn):
    font = get_font()
    surf = font.render(s, True, color)
    rect = surf.get_rect(**posn)
    return surf, rect
import sys, pygame
import io
import cairosvg

def load_svg(path, size=None):
    if size is None:
        w, h = None, None
    else:
        w, h = size
    png_bytes = cairosvg.svg2png(url=path, output_width=130, output_height=182)
    return pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
    
pygame.init()

size = width, height = 800, 600
fps = 60

speed = [60/800, 60/800]
green = (85, 170, 85)

screen = pygame.display.set_mode(size)

ball = load_svg("assets/full-deck/KC.svg")
ballrect = ball.get_rect()
paused = False

clock = pygame.time.Clock()

while True:
    dt = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    
    keys = pygame.key.get_pressed()
    if not keys[pygame.K_SPACE]:
        ballrect = ballrect.move([s * dt for s in speed])
        if ballrect.left < 0 or ballrect.right > width:
            speed[0] = -speed[0]
        if ballrect.top < 0 or ballrect.bottom > height:
            speed[1] = -speed[1]

        screen.fill(green)
        screen.blit(ball, ballrect)
        pygame.display.flip()
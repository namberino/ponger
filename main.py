import pygame
from pong import Game

class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_game(self):
        clk = pygame.time.Clock()
        run = True

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()

            # left side control
            if keys[pygame.K_w]:
                game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                game.move_paddle(left=True, up=False)

            # right side control
            if keys[pygame.K_UP]:
                game.move_paddle(left=False, up=True)
            if keys[pygame.K_DOWN]:
                game.move_paddle(left=False, up=False)

            game.loop()
            game.draw()

            clk.tick(60)
            pygame.display.update()
            
        pygame.quit()

WIN_WIDTH, WIN_HEIGHT = 700, 500
window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
game = Game(window, WIN_WIDTH, WIN_HEIGHT)

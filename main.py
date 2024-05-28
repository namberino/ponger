import pygame
from pong import Game
import neat
import os
import pickle

class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

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
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            # right side control
            out = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision = out.index(max(out))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw()

            clk.tick(60)
            pygame.display.update()
            
        pygame.quit()

    def calc_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            out1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            decision1 = out1.index(max(out1))

            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            out2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision2 = out2.index(max(out2))

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            #print(decision1, decision2)
            
            game_info = self.game.loop()
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calc_fitness(genome1, genome2, game_info)
                break

# train the AIs against every other AI and get the sum of fitness
def evaluate_genome_fitness(genomes, config):
    WIN_WIDTH, WIN_HEIGHT = 700, 500
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    # each genome will play against other genomes
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break

        genome1.fitness = 0 # no fitness by default

        for genome_id2, genome2 in genomes[i + 1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window, WIN_WIDTH, WIN_HEIGHT)

            game.train_ai(genome1, genome2, config)

def run_neat(config):
    ppl = neat.Checkpointer.restore_checkpoint('neat-checkpoint-6') # loading checkpoint
    #ppl = neat.Population(config) # initializing population of genomes
    ppl.add_reporter(neat.StdOutReporter(True)) # report data to stdout
    stats = neat.StatisticsReporter()
    ppl.add_reporter(stats)
    ppl.add_reporter(neat.Checkpointer(1)) # save checkpoint after n generation

    best = ppl.run(evaluate_genome_fitness, 1) # run for 50 generations

    with open("best.pickle", "wb") as f:
        pickle.dump(best, f)

def test_best_ai(config):
    WIN_WIDTH, WIN_HEIGHT = 700, 500
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    with open("best.pickle", "rb") as f:
        best = pickle.load(f)

    game = PongGame(window, WIN_WIDTH, WIN_HEIGHT)
    game.test_ai(best, config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    #run_neat(config)
    test_best_ai(config)

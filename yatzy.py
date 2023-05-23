


ROUNDS = ['Ones',
          'Twos', 
          'Threes', 
          'Fours', 
          'Fives', 
          'Sixes',
          'One pair',
          'Two pairs',
          '3 of a kind',
          '4 of a kind', 
          'Full house', 
          'Low straight', 
          'High straight', 
          'Chance', 
          'Yatzee']

class Player:
    def __init__(self, name):
        self.name = name
        self.board = [ None for _ in range(len(ROUNDS)) ]
        self.bonus = 0

    def insert_score(self, score):
        self.board[self.board.index(None)] = score

    def calc_bonus(self):
        s = sum(self.board[n] if self.board[n] is not None else 0 for n in range(6))
        if s >= 42:
            self.bonus = 50
        else:
            self.bonus = 0

    def get_total_score(self):
        return sum( x if x is not None else 0 for x in self.board) + self.bonus

class Game:
    def __init__(self, num_players=2):
        self.players = [Player('Player3'), Player('Player2')]
        
        self.round = 0
        self.player_select = 0
        self.done = False

    def __str__(self):
        size = 15
        s = f'{"Round": ^{size}}|'

        for p in self.players:
            s += f'{p.name: ^{size}}|'

        #First section
        for r in range(6):
            s += '\n' + '-'*(size*3+3) + '\n'
            s += f'{ROUNDS[r]: <{size}}|'
            for p in self.players:
                if p.board[r] is None:
                    s += ' '*size + '|'
                else:
                    s += f'{str(p.board[r]): ^{size}}|'
        
        #bonus
        s += '\n' + '='*(size*3+3) + '\n'
        s += f'{"Bonus": ^{size}}|'
        for p in self.players:
            s += f'{p.bonus: ^{size}}'
        s += '\n' + '='*(size*3+3) + '\n'
        
        #Second section
        for r in range(6, len(ROUNDS)):
            s += f'{ROUNDS[r]: <{size}}|'
            for p in self.players:
                if p.board[r] is None:
                    s += ' '*size + '|'
                else:
                    s += f'{str(p.board[r]): ^{size}}|'
            s += '\n' + '-'*(size*3+3) + '\n'


        s += f'{"TOTAL": ^{size}}|'
        for p in self.players:
            s += f'{p.get_total_score(): ^{size}}|'


        if self.done:
            res = [ p.get_total_score() for p in self.players ]
            res_sort = sorted(res, reverse=True)

            s += f'\n\n{self.players[res.index(max(res))].name} is the winner!!\nThe margin to the losers was more than {res_sort[0] - res_sort[1]}'
            

        return s

    def dice_roll(self, dices):
        score = self.calculate_score(dices, self.round)
        

        self.players[self.player_select].insert_score(score)
        self.player_select = (self.player_select + 1) % len(self.players)
        self.players[self.player_select].calc_bonus()
        if self.player_select == 0:
            self.round += 1

        if self.round == len(ROUNDS) - 1 and self.player_select == len(self.players) - 1:
            self.done = True

    def get_current_player(self):
        return self.players[self.player_select].name
    
    def calculate_score(self, dices, round):
        dices.sort()

        if round <= 5:
            return dices.count(round + 1) * (round + 1)
        elif round == 8: #3 of a kind
            for n in set(dices):
                if dices.count(n) >= 3:
                    return n*3
            return 0
        elif round == 6: #One pair
            dices.sort(reverse=True)
            for n in dices:
                if dices.count(n) >= 2:
                    return 2*n 
            return 0
        elif round == 7: #Two pairs
            one = 0
            for n in set(dices):
                if dices.count(n) >= 2 and one == 0:
                    one = 2*n
                elif dices.count(n) >= 2:
                    return one + 2*n
            return 0
        elif round == 8: #3 of a kind
            for n in set(dices):
                if dices.count(n) >= 3:
                    return n*3
            return 0
        elif round == 9: #4 of a kind
            for n in set(dices):
                if dices.count(n) >= 4:
                    return n*4
            return 0
        elif round == 10: #Full house
            if len(set(dices)) <= 2 and (dices.count(dices[0]) == 2 or dices.count(dices[0]) == 3 or dices.count(dices[0]) == 5):
                return sum(dices)
            return 0
        elif round == 11: #Low straight
            if len(set(dices)) == 5 and dices[0] == 1:
                return sum(dices)
            return 0
        elif round == 12: #High straight
            if len(set(dices)) == 5 and dices[0] == 2:
                return sum(dices)
            return 0
        elif round == 13: #Chance
            return sum(dices)
        elif round == 14: #Yatzee
            if len(set(dices)) == 1:
                return 50
            return 0
        else:
            raise Exception(f'Round number is not valid, {round}')



def test_calculate_score(dices, round, res):
    game = Game()
    r = game.calculate_score(dices, round)
    assert r == res, f'Expected {res}, got {r}. Dices: {dices}, round: "{ROUNDS[round]}"'

if __name__ == '__main__':
    print('Testing calculate_score():')
    test_calculate_score([1, 1, 1, 3, 5], 0, 3)
    test_calculate_score([1, 1, 1, 3, 5], 1, 0)
    test_calculate_score([1, 1, 1, 3, 5], 2, 3)
    test_calculate_score([3, 1, 3, 3, 5], 2, 9)
    test_calculate_score([1, 6, 1, 6, 5], 5, 12)

    #One pair
    test_calculate_score([1, 1, 1, 1, 1], 6, 2)
    test_calculate_score([3, 1, 3, 3, 5], 6, 6)
    test_calculate_score([1, 6, 2, 4, 5], 6, 0)
    test_calculate_score([1, 6, 4, 4, 6], 6, 12)
    
    #Two pairs
    test_calculate_score([1, 1, 1, 1, 1], 7, 0)
    test_calculate_score([3, 1, 3, 3, 5], 7, 0)
    test_calculate_score([6, 6, 2, 4, 4], 7, 20)
    test_calculate_score([6, 6, 6, 4, 4], 7, 20)
    test_calculate_score([6, 6, 6, 6, 4], 7, 0)
    test_calculate_score([6, 6, 6, 6, 6], 7, 0)
    
    test_calculate_score([1, 1, 1, 3, 5], 8, 3)
    test_calculate_score([1, 1, 2, 3, 5], 8, 0)
    test_calculate_score([3, 1, 3, 3, 5], 8, 9)
    test_calculate_score([1, 5, 5, 5, 5], 8, 15)
    test_calculate_score([1, 1, 2, 3, 6], 8, 0)
    
    test_calculate_score([1, 1, 1, 1, 5], 9, 4)
    test_calculate_score([1, 1, 2, 3, 5], 9, 0)
    test_calculate_score([3, 1, 3, 3, 5], 9, 0)
    test_calculate_score([5, 1, 5, 5, 5], 9, 20)
    test_calculate_score([6, 6, 6, 6, 6], 9, 24)
    
    test_calculate_score([1, 1, 5, 5, 5], 10, 17)
    test_calculate_score([1, 1, 2, 3, 5], 10, 0)
    test_calculate_score([3, 3, 3, 5, 5], 10, 19)
    test_calculate_score([5, 1, 5, 5, 5], 10, 0)
    test_calculate_score([6, 6, 6, 6, 6], 10, 30)
    
    test_calculate_score([1, 2, 3, 4, 5], 11, 15)
    test_calculate_score([5, 1, 2, 3, 4], 11, 15)
    test_calculate_score([6, 6, 6, 6, 6], 11, 0)
    test_calculate_score([2, 3, 4, 5, 6], 11, 0)
    

    test_calculate_score([2, 3, 4, 5, 6], 12, 20)
    test_calculate_score([6, 3, 2, 5, 4], 12, 20)
    test_calculate_score([6, 6, 6, 6, 6], 12, 0)
    test_calculate_score([6, 6, 1, 4, 6], 12, 0)
    test_calculate_score([1, 2, 3, 4, 5], 12, 0)
    
    test_calculate_score([1, 3, 3, 4, 5], 13, 16)
    test_calculate_score([1, 2, 3, 4, 5], 14, 0)
    test_calculate_score([1, 1, 1, 1, 1], 14, 50)
    test_calculate_score([6, 6, 6, 6, 6], 14, 50)

    print('All tests are done')

    game = Game()
    print(game)

    game.dice_roll([1, 1, 1, 2, 4])
    game.dice_roll([1, 1, 1, 2, 4])
    game.dice_roll([1, 1, 1, 2, 4])

    
    game.dice_roll([1, 1, 1, 2, 4])
    game.dice_roll([1, 2, 1, 2, 4])
    game.dice_roll([1, 4, 6, 2, 4])

    game.dice_roll([1, 1, 1, 2, 4])
    game.dice_roll([1, 2, 1, 2, 4])
    game.dice_roll([1, 4, 6, 2, 4])
    
    game.dice_roll([1, 1, 1, 2, 4])
    game.dice_roll([1, 2, 1, 2, 4])
    game.dice_roll([1, 2, 2, 2, 4])

    print(game)

    


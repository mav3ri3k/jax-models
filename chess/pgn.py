import chess.pgn
import chess
import chess.engine
from joblib import Parallel, delayed


# engine = chess.engine.SimpleEngine.popen_uci(r"/usr/local/bin/stockfish")
pgn = open("./lichess_db_standard_rated_2014-07.pgn")

# offsets = []
# while True:
#     offset = pgn.tell()
#     headers = chess.pgn.read_headers(pgn)
#     if headers is None:
#         break
#     offsets.append(offset)
# print(offsets)

rating = [0, 0, 0, 0, 0, 0]

total = 0
while True:
    headers = chess.pgn.read_headers(pgn)

    if headers == None:
        break

    white = headers['WhiteElo']
    black = headers['BlackElo']

    if white == '?' or black == '?':
        continue
    elo = (int(white) + int(black)) // 2

    if elo <= 500:
        rating[0] += 1
    elif elo <= 1000:
        rating[1] += 1
    elif elo <= 1500:
        rating[2] += 1
    elif elo <= 2000:
        rating[3] += 1
    elif elo <= 2500:
        rating[4] += 1
    else:
        rating[5] += 1

    total += 1

elo_level = 500
for i in rating:
    print(elo_level, i, i/total)
    elo_level += 500

# game =  chess.pgn.read_game(pgn)


# while (game != None):
#     game = chess.pgn.read_game(pgn)
#     i += 1

# engine.quit()


"""
board = game.board()

i = 0
for move in game.mainline_moves():
    i += 1
    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(move)
    print(f"{i} Actual: {move}, Engine: {result.move}")

    if(i == 60):
        print(board)

"""

"""
while (game != None):
    game = chess.pgn.read_game(pgn)
    i += 1
"""

"""
first_game = chess.pgn.read_game(pgn)

second_game = chess.pgn.read_game(pgn)

print(first_game.headers["Event"])

board = first_game.board()

for move in first_game.mainline_moves():
    board.push(move)
    print(board)
    print()

print(board)
"""

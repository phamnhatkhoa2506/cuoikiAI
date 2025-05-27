import math

# Kích thước bàn cờ
SIZE = 5
EMPTY = '-'
PLAYER = 'X'
AI = 'O'
MAX_DEPTH = 2  # Giới hạn độ sâu tìm kiếm

def create_board():
    return [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

def is_winner(board, player):
    # Kiểm tra hàng và cột
    for i in range(SIZE):
        if all(board[i][j] == player for j in range(SIZE)) or all(board[j][i] == player for j in range(SIZE)):
            return True
    # Kiểm tra đường chéo chính
    if all(board[i][i] == player for i in range(SIZE)):
        return True
    # Kiểm tra đường chéo phụ
    if all(board[i][SIZE - 1 - i] == player for i in range(SIZE)):
        return True
    return False

def is_full(board):
    return all(board[i][j] != EMPTY for i in range(SIZE) for j in range(SIZE))

def minimax(board, depth, is_maximizing):
    if is_winner(board, AI):
        return 10 - depth
    if is_winner(board, PLAYER):
        return depth - 10
    if is_full(board) or depth >= MAX_DEPTH:
        return 0
    
    if is_maximizing:
        best_score = -math.inf
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == EMPTY:
                    board[i][j] = AI
                    score = minimax(board, depth + 1, False)
                    board[i][j] = EMPTY
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == EMPTY:
                    board[i][j] = PLAYER
                    score = minimax(board, depth + 1, True)
                    board[i][j] = EMPTY
                    best_score = min(best_score, score)
        return best_score

def best_move(board):
    best_score = -math.inf
    move = None
    for i in range(SIZE):
        for j in range(SIZE):
            if board[i][j] == EMPTY:
                board[i][j] = AI
                score = minimax(board, 0, False)
                board[i][j] = EMPTY
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def play_game():
    board = create_board()
    print_board(board)
    
    while True:
        # Người chơi đánh
        x, y = map(int, input("Nhập vị trí (hàng cột): ").split())
        if board[x][y] != EMPTY:
            print("Ô đã được chọn, hãy chọn ô khác!")
            continue
        board[x][y] = PLAYER
        print_board(board)
        if is_winner(board, PLAYER):
            print("Người chơi thắng!")
            break
        if is_full(board):
            print("Hòa!")
            break
        
        # AI đánh
        print("Lượt AI...")
        move = best_move(board)
        if move:
            board[move[0]][move[1]] = AI
        print_board(board)
        if is_winner(board, AI):
            print("AI thắng!")
            break
        if is_full(board):
            print("Hòa!")
            break

play_game()

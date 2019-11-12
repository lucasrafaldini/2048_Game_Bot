from PIL import ImageOps
import pyscreenshot as ImageGrab
import pyautogui
import time

UP = 100
LEFT = 101
DOWN = 102
RIGHT = 103

scoreGrid = [50, 30, 15, 5,
             0, -10, 0, 0,
             0, 0, 0, 0,
             5, 0, 0, 0]


class Cords:
    cord11 = (170, 270)
    cord12 = (270, 270)
    cord13 = (370, 270)
    cord14 = (470, 270)
    cord21 = (170, 370)
    cord22 = (270, 370)
    cord23 = (370, 370)
    cord24 = (470, 370)
    cord31 = (170, 480)
    cord32 = (270, 480)
    cord33 = (370, 480)
    cord34 = (470, 480)
    cord41 = (170, 590)
    cord42 = (270, 590)
    cord43 = (370, 590)
    cord44 = (470, 590)

    cordArray = [cord11, cord12, cord13, cord14,
                 cord21, cord22, cord23, cord24,
                 cord31, cord32, cord33, cord34,
                 cord41, cord42, cord43, cord44]


def getGrid():
    image = ImageGrab.grab()
    grayImage = ImageOps.grayscale(image)
    for index, cord in enumerate(Cords.cordArray):
        pixel = grayImage.getpixel(cord)
        pos = Values.valueArray.index(pixel)
        if pos == 0:
           currentGrid[index] = 0
        else:
            currentGrid[index] = pow(2, pos)

class Values:
    empty = 195
    two = 229
    four = 225
    eight = 190
    sixteen = 172
    thirtyTwo = 157
    sixtyFour = 135
    oneTwentyEight = 205
    twoFiftySix = 201
    fiveOneTwo = 197
    oneZeroTwoFour = 193
    twoZeroFourEight = 189

    valueArray = [empty, two, four, eight, sixteen, thirtyTwo, sixtyFour,
                  oneTwentyEight, twoFiftySix, fiveOneTwo, oneZeroTwoFour,
                  twoZeroFourEight]


currentGrid = [0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0,
               0, 0, 0, 0]


def printGrid(grid):
    for i in range(16):
        if i % 4 == 0:
            print("[ " + str(grid[i]) + " " + "]" + "[ " + str(grid[i + 1]) + " " + "]" + "[ " + str(
                grid[i + 2]) + " " + "]" + "[ " + str(grid[i + 3]) + " " + "]")


def swipeRow(row):
    prev = -1  # previous non-zero element
    i = 0
    temp = [0, 0, 0, 0]

    for element in row:
        if element != 0:
            if prev == -1:
                prev = element
                temp[i] = element
                i += 1
            elif prev == element:
                temp[i - 1] = 2 * prev
                prev = -1
            else:
                prev = element
                temp[i] = element
                i += 1
    return temp


def getNextGrid(grid, move):
    temp = [0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0]
    if move == UP:
        for i in range(4):
            row = []
            for j in range(4):
                row.append(grid[i + 4 * j])
            row = swipeRow(row)
            for j, val in enumerate(row):
                temp[i + 4 * j] = val
    elif move == LEFT:
        for i in range(4):
            row = []
            for j in range(4):
                row.append(grid[4 * i + j])
            row = swipeRow(row)
            for j, val in enumerate(row):
                temp[4 * i + j] = val
    elif move == DOWN:
        for i in range(4):
            row = []
            for j in range(4):
                row.append(grid[i + 4 * (3 - j)])
            row = swipeRow(row)
            for j, val in enumerate(row):
                temp[i + 4 * (3 - j)] = val
    elif move == RIGHT:
        for i in range(4):
            row = []
            for j in range(4):
                row.append(grid[4 * i + (3 - j)])
            row = swipeRow(row)
            for j, val in enumerate(row):
                temp[4 * i + (3 - j)] = val

    return temp


def getScore(grid):
    score = 0
    for i in range(4):
        for j in range(4):
            score += grid[4 * i + j] * scoreGrid[4 * i + j]
    return score

def getBestMove(grid):
    temp = [0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0]

    scoreUp = getNextGrid(grid, UP)
    scoreDown = getNextGrid(grid, DOWN)
    scoreLeft = getNextGrid(grid, LEFT)
    scoreRight = getNextGrid(grid, RIGHT)

    if not isMoveValid(grid, UP):
        #scoreUp = list(reversed(scoreGrid))
        scoreUp = temp
    if not isMoveValid(grid, DOWN):
        #scoreDown = list(reversed(scoreGrid))
        scoreDown = temp
    if not isMoveValid(grid, LEFT):
        #scoreLeft = list(reversed(scoreGrid))
        scoreLeft = temp
    if not isMoveValid(grid, RIGHT):
        #scoreRight = list(reversed(scoreGrid))
        scoreRight = temp

    maxScore = max(scoreUp,scoreDown,scoreLeft,scoreRight)

    if scoreUp == maxScore:
        return UP
    elif scoreDown == maxScore:
        return DOWN
    elif scoreLeft == maxScore:
        return LEFT
    elif scoreRight == maxScore:
        return RIGHT

def isMoveValid(grid, move):
    if getNextGrid(grid, move) == grid:
        return False
    else:
        return True

def performMove(move):
    if move == UP:
        pyautogui.keyDown('up')
        print("UP")
        time.sleep(0.1)
        pyautogui.keyUp('up')
    elif move == DOWN:
        pyautogui.keyDown('down')
        print("DOWN")
        time.sleep(0.1)
        pyautogui.keyUp('down')
    elif move == LEFT:
        pyautogui.keyDown('left')
        print("LEFT")
        time.sleep(0.1)
        pyautogui.keyUp('left')
    else:
        pyautogui.keyDown('right')
        print("RIGHT")
        time.sleep(0.1)
        pyautogui.keyUp('right')


def main():
    time.sleep(1)
    while True:
        getGrid()
        performMove(getBestMove(currentGrid))
        time.sleep(0.1)


if __name__ == '__main__':
    main()

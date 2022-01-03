import nltk
# nltk.download('wordnet')
from cheating import *  # lazy import so I can write cleaner code here
import pyautogui, pygetwindow
import mss
from multiprocessing.pool import Pool, ThreadPool


def show_board(board):
    for y in range(len(board[0])):
        for x in range(len(board)):
            print(board[x][y], end="")
        print("")


def show_grid(img, xarr, yarr):
    img2 = img.copy()  # to show the grid for debugging
    il = img2.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if x in xarr or y in yarr:
                il[x, y] = (0, 255, 255)
    img2.show()


def dostuff():
    start = time.time()
    first_start = start
    sct = mss.mss()

    z1 = pygetwindow.getAllTitles()
    wind = ""
    for z in z1:
        if "To the king goes his crown" in z:  # change this to  the appropriate window name
            wind = z
            break
            # print("yes")

    my = pygetwindow.getWindowsWithTitle(wind)[0]
    # my.activate()
    s = my.size
    p = my.box
    area = {"top":p.top, "left":p.left, "width":p.width, "height":p.height}

    imag = sct.grab(area)
    test = Image.frombytes("RGB", (imag.size[0], imag.size[1]), imag.rgb)

    test = Image.open("test/test12.jpg")  # static image screenshots for debugging

    print(test.size)
    var = 1200 / test.size[1]  # images larger than a certain size just get harder to process for no real accuracy benefit
    # var = 1.3
    img = test
    if var > 1:
        img = img.resize((round(test.size[0] / var), round(test.size[1] / var)), Image.BICUBIC)
    # img = test

    print("game image gotten...", time.time() - start)
    start = time.time()

    # hand_result = thr.apply_async(get_hand, [img.copy()])

    # print("2", time.time() - start)
    # start = time.time()

    xarr, yarr = get_grid(img, 3)
    # show_grid(img, xarr, yarr)

    print("board grid gotten...", time.time() - start)
    start = time.time()

    board_stats = get_board(xarr, yarr, img)
    board = board_stats[1]

    print("board tiles gotten...", time.time() - start)
    start = time.time()

    # hand = hand_result.get()

    hand = get_hand(img.copy())

    print("hand tiles gotten...", time.time() - start)
    start = time.time()

    move_sets = get_possible(hand, *board_stats)

    print("valid words filtered...", time.time() - start)
    print("total time:", time.time() - first_start)

    show_board(board)

    print("Owned:")
    print(board_stats[0])

    print("Tiles in hand:")
    print(hand)

    # valid_words = [*{*valid_words}]

    if len(move_sets) > 0:
        print("Possible words:")
        for i in move_sets:
            valid_words = i
            numb = math.ceil(len(valid_words) / 10)
            for i in range(numb):
                print(", ".join(valid_words[i*15:(i+1)*15]))
    else:
        print("No words predicted with currently available information...")


if __name__ == "__main__":
    # start = time.time()

    # directory = "test/"
    #
    # img = Image.open(directory + "screen.png").convert("RGB")

    # thr = Pool(processes=1)

    # t_arr = []
    dostuff()
    # while True:
    #     t_arr.append(thr.apply_async(dostuff))
    #     time.sleep(2)
    #     dum = t_arr.pop(0)
    #     dum.get()
    #     break

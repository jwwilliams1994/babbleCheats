from cheating import *  # lazy import so I can write cleaner code here
import pyautogui, pygetwindow
import mss


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
    sct = mss.mss()

    z1 = pygetwindow.getAllTitles()
    wind = ""
    for z in z1:
        if "To the king goes his crown" in z:
            wind = z
            # print("yes")

    my = pygetwindow.getWindowsWithTitle(wind)[0]
    # my.activate()
    s = my.size
    p = my.box
    area = {"top":p.top, "left":p.left, "width":p.width, "height":p.height}

    imag = sct.grab(area)
    test = Image.frombytes("RGB", (imag.size[0], imag.size[1]), imag.rgb)
    var = 1200 / test.size[1]
    img = test
    if var > 1:
        img = img.resize((round(test.size[0] / var), round(test.size[1] / var)), Image.BICUBIC)
    # img = test

    xarr, yarr = get_grid(img, 3)
    # show_grid(img, xarr, yarr)

    print(time.time() - start)
    start = time.time()

    board_stats = get_board(xarr, yarr, img)
    board = board_stats[2]

    print(time.time() - start)
    start = time.time()

    valid_words = get_possible(*board_stats)

    print(time.time() - start)
    show_board(board)
    print("Tiles in hand:")
    print(board_stats[0])

    if len(valid_words) > 0:
        print("Possible words:")
        print(", ".join(valid_words))
    else:
        print("No words predicted with currently available information...")


if __name__ == "__main__":
    # start = time.time()

    # directory = "test/"
    #
    # img = Image.open(directory + "screen.png").convert("RGB")

    thr = Pool(processes=6)

    t_arr = []
    while True:
        t_arr.append(thr.apply_async(dostuff))
        time.sleep(2)
        dum = t_arr.pop(0)
        dum.get()
        break

import math
import re
import string
import time
import numpy as np
import cv2 as cv
from mss import mss
from PIL import Image, ImageFilter, ImageChops, ImageOps
import colorsys
import os
import random
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import nltk
from nltk.corpus import words as wd
from nltk.corpus import wordnet
from multiprocessing.pool import Pool, ThreadPool
from functools import partial

let_dir = "letters/"
letter_files = os.listdir(os.getcwd() + "/" + let_dir)
wordset = set(wd.words())  # simply, a dictionary of all possible words
p = nltk.PorterStemmer()


def dist(r, g, b, col=(118, 103, 80), v1=0.1, v2=0.15, v3=0.15):  # default col is average grid color
    col = colorsys.rgb_to_hsv(col[0] / 255, col[1] / 255, col[2] / 255)
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if abs(h - col[0]) < 0.1:
        if abs(s - col[1]) < 0.15:
            if abs(v - col[2]) < 0.15:
                return True
    return False


def simplify(inp, inp2, weighted=False):
    out_arr = []
    if weighted:
        arr = [inp[0] * inp2[0]]
    else:
        arr = [inp[0]]
    arr2 = [inp2[0]]
    if weighted:
        for i in range(1, len(inp)):
            if inp[i] - inp[i - 1] > 1:
                out = sum(arr) / sum(arr2)
                out_arr.append(out)
                arr = [inp[i] * inp2[i]]
                arr2 = [inp2[i]]
            else:
                arr.append(inp[i] * inp2[i])
                arr2.append(inp2[i])
    else:
        for i in range(1, len(inp)):
            if inp[i] - inp[i - 1] > 1:
                out = sum(arr) / len(arr)
                out_arr.append(out)
                arr = [inp[i]]
            else:
                arr.append(inp[i])
    return out_arr


def solve2(inp):
    # arr = [a - inp[0] for a in inp][1:]
    offset = inp[0]
    arr = list(map(lambda a: a - offset, inp))[1:]
    m_arr = []
    # for i in range(1, len(arr)):
    #     m_arr.append(arr[i] - arr[i - 1])
    m_arr = list(map(lambda i: arr[i] - arr[i - 1], range(1, len(arr))))
    print(m_arr)

    for i in range(1, len(arr)-1):
        diff = m_arr[i - 1] - m_arr[i]
        if diff > 2:
            for r in range(2, 6):
                if r + 0.1 > m_arr[i - 1] / m_arr[i] > r - 0.1:
                    m_arr[i - 1] = m_arr[i - 1] / r

    m_arr.sort()
    m_arr2 = [m_arr[1]]
    for i in range(1, len(m_arr)):
        diff = m_arr[i] - m_arr[i - 1]
        if diff > 1:
            break
        m_arr2.append(m_arr[i])
    m = sum(m_arr2) / len(m_arr2)
    n = round(arr[-1] / m)
    m = arr[-1] / n
    while offset > m:
        offset -= m
    return m, offset


def get_grid(inp, val=1):
    w, h = inp.size
    inp = inp.resize((round(w / val), round(h / val)), Image.NEAREST)
    w, h = inp.size
    img1 = inp.convert("RGB").copy()
    img2 = inp.convert("L").copy()

    il = img1.load()
    il2 = img2.load()
    for x in range(w):
        for y in range(h):
            r, g, b = il[x, y]
            if dist(r, g, b):
                il2[x, y] = 255
            else:
                il2[x, y] = 0

    w, h = img2.size
    x_arr = []
    y_arr = []
    for x in range(w):
        count = 0
        for y in range(h):
            if il2[x, y] == 255:
                count += 1
        x_arr.append([count, x])

    for y in range(h):
        count = 0
        for x in range(w):
            if il2[x, y] == 255:
                count += 1
        y_arr.append([count, y])

    # plotty(x_arr, y_arr, 98)

    x_lim = np.percentile([*(a[0] for a in x_arr)], 98)  # to trim unwanted noise
    y_lim = np.percentile([*(a[0] for a in y_arr)], 98)
    # print(x_lim, y_lim)
    x_arr = [*filter(lambda a: a[0] > x_lim, x_arr)]
    y_arr = [*filter(lambda a: a[0] > y_lim, y_arr)]

    nx_arr = [*(a[0] for a in x_arr)]
    x_arr = [*(a[1] for a in x_arr)]

    ny_arr = [*(a[0] for a in y_arr)]
    y_arr = [*(a[1] for a in y_arr)]

    xm, xb = solve2(simplify(x_arr, nx_arr))
    ym, yb = solve2(simplify(y_arr, ny_arr))

    if abs(xm - ym) > 0.5:
        if xm < ym:
            ym = xm
        else:
            xm = ym

    xout_arr = []
    xb = xb * (1 + ((val - 1) * 0.02))
    yb = yb * (1 + ((val - 1) * 0.02))
    for i in range(round(inp.size[0] / xm)):
        result = round((xm * i + xb) * val)
        if result > (inp.size[0] * val):
            break
        else:
            xout_arr.append(result)

    yout_arr = []
    for i in range(round(inp.size[1] / ym)):
        result = round((ym * i + yb) * val)
        if result > (inp.size[0] * val):
            break
        else:
            yout_arr.append(result)
    return xout_arr, yout_arr


def most_similar(im, im_list):
    sim_arr = []
    for i in im_list:
        im2 = np.asarray(i)
        sim_arr.append(round(ssim(im, im2), 3))
    maxi = max(sim_arr)
    if maxi < 0.8:
        return ord("?") - 65
    ind = sim_arr.index(maxi)
    return ind


def get_similarity(img, img2):
    img2 = np.asarray(img2)
    out = round(mean_squared_error(img, img2), 3)
    return out


def get_sim(img, img2):  # this one is currently used, above and below are ignored
    img2 = np.asarray(img2)
    res = cv.matchTemplate(img, img2, cv.TM_SQDIFF_NORMED)
    # print(cv.minMaxLoc(res))
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return min_val


def get_dist(img, img2):
    img2 = np.asarray(img2)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    d_arr = list(map(lambda a: a[0].distance, matches))
    # d_arr = [a[0].distance for a in matches]
    return min(d_arr)


def sieve(img, letters):  # letter matching similar can be tricky...
    if "Q" in letters and "O" in letters:
        img = img.crop((13, 19, 25, 25))
    if "G" in letters and ("Q" in letters or "O" in letters):
        img = img.crop((20, 10, 25, 20))
    if "E" in letters and "F" in letters:
        img = img.crop((0, 16, 25, 25))
    if "E" in letters and "L" in letters:
        img = img.crop((0, 0, 25, 20))
    if "B" in letters and "D" in letters:
        img = img.crop((0, 8, 25, 17))
    if "R" in letters and "P" in letters:
        img = img.crop((20, 20, 25, 25))
    if "L" in letters and "U" in letters:
        img = img.crop((20, 0, 25, 25))
    if "O" in letters and "D" in letters:
        img = img.crop((0, 0, 5, 5))
    if "H" in letters and "R" in letters:
        img = img.crop((0, 0, 25, 10))
    return img


def most_similar2(im, im_list, diff=False, debug=False):
    im2 = im.copy()
    im = np.asarray(im)
    if diff:
        arr = list(map(lambda a: round(get_sim(im, a), 3), im_list))
        # arr = [round(get_sim(im, a), 3) for a in im_list]
        arr.sort()
        return arr[0]
    # print(chr(most_similar(im, im_list) + 65), end=" : ")
    sim_arr = []
    o_arr = [*zip(map(lambda a: get_sim(im, a), im_list), list(string.ascii_uppercase))]
    # o_arr = [*zip([get_sim(im, a) for a in im_list], list(string.ascii_uppercase))]
    o_arr.sort()
    if debug:
        print(o_arr)
    letters = [o_arr[0][1], o_arr[1][1]]  # to make doubly sure it's what it's supposed to be, the matching can be otherwise drunk with similar characters
    o_arr = [*zip(map(lambda a: (get_sim(np.asarray(sieve(im2, letters)), sieve(a, letters))), im_list), list(string.ascii_uppercase))]
    # o_arr = [*zip([get_sim(np.asarray(sieve(im2, letters)), sieve(a, letters)) for a in im_list], list(string.ascii_uppercase))]
    o_arr = list(filter(lambda a: a[1] in letters, o_arr))
    o_arr.sort()
    num = ord(o_arr[0][1]) - 65
    return num


def normalize(img, debug=False):
    # img = Image.eval(img, (lambda x: (x > 200) * x))
    w, h = img.size
    il = img.load()
    l, u, d, r = w, w, 0, 0
    for x in range(w):
        for y in range(h):
            if il[x, y] >= 220:
                if x < l:
                    l = x
                if y < u:
                    u = y
                if x > r:
                    r = x
                if y > d:
                    d = y
    img = img.crop((l, u, r, d))
    img = img.resize((25, 25), Image.BICUBIC)
    img = Image.eval(img, (lambda a: (a > 225) * 255))
    if debug:
        img.show()
    return img


def check_teal(img):
    col = (114, 152, 225)
    im_iter = img.getdata()
    count = 0
    count = sum(map(lambda i: dist(i[0], i[1], i[2], col), im_iter))
    perc = round(count / (img.size[0] * img.size[1]), 3)
    return perc


def check_tan(img):
    col = (255, 246, 167)
    im_iter = img.getdata()
    count = 0
    count = sum(map(lambda i: dist(i[0], i[1], i[2], col), im_iter))
    perc = round(count / (img.size[0] * img.size[1]), 3)
    return perc


def check_hand_color(img):
    col = (77, 111, 111)
    im_iter = img.getdata()
    count = 0
    count = sum(map(lambda i: dist(i[0], i[1], i[2], col), im_iter))
    perc = round(count / (img.size[0] * img.size[1]), 3)
    return perc


def get_letter(img, debug=False):  # img needs to be normalized first
    im_arr = list(map(lambda i: Image.open(let_dir + i).convert("L"), letter_files))
    # im_arr = [Image.open(let_dir + i).convert("L") for i in letter_files]
    num = most_similar2(img, im_arr)
    return chr(num + 65)


def get_hand(img, debug=False):
    w, h = img.size
    il = img.load()
    col = (77, 111, 111)
    x = round(w / 2)
    st = 0
    en = 0
    for y in range(h):
        r, g, b = il[x, y]
        if dist(r, g, b, col):
            if st == 0:
                st = y
            en = y
    yrat1 = (st / h) + 0.03
    yrat2 = (en / h) - 0.03
    # print(st, en)
    midy = ((h * yrat2) + (h * yrat1)) / 2
    st = 0
    en = 0
    for x in range(w):
        r, g, b = il[x, midy]
        if dist(r, g, b, col):
            if st == 0:
                st = x
            en = x
    xrat1 = (st / w) + 0.01
    xrat2 = (en / w) - 0.014
    # print(st, en)
    start = (round(xrat1 * w), round(yrat1 * h))
    end = (round(xrat2 * w), round(yrat2 * h))
    img = img.crop((*start, *end)).convert("L")  # crops out the area the hand tiles exist in, no reason to bother looking elsewhere
    # img.save('hand.png')
    img = Image.eval(img, (lambda x: (x < 20) * 255))
    # img.show()
    w, h = img.size
    port = w / 10
    hand_arr = []
    for i in range(10):
        area = (round(port * i), 0, round(port * (i + 1)), h)
        out = img.crop(area)
        if debug:
            out.save(str(i) + ".png")
        w, h = out.size
        area = (round(.2 * w), round(.1 * h), round(.8 * w), round(.8 * h))
        out = out.crop(area)
        hist = out.histogram()[255] / (out.size[0] * out.size[1])  # percentage of pixels that are full white
        if hist > 0.02:
            out = get_letter(normalize(out), True)
            hand_arr.append(out)
    return hand_arr


def get_possible(hand, owned, board):
    # hand = [a.lower() for a in hand]
    hand = list(map(str.lower, hand))
    char_list = list(string.ascii_lowercase)
    build_arr = []

    def get_matching(inp, debug=False):  # gets possible words that match input regex, still needs board adjacency validation afterwards
        base = "".join(map(lambda a: a[2], owned))
        lim = len(inp)
        inpj = "".join(inp).lower()
        inpl = list(map(str.lower, inp))
        conj = [inpj, "".join(hand)]
        in_dict = {*inpj.replace("_", "")}
        inplr = inpl[::-1]
        last = len(inp) - min(map(inplr.index, in_dict))
        # last = len(inp) - min([inpl[::-1].index(a) for a in in_dict])
        break_points = []

        origin = 0
        for i in range(len(inp)):
            if inp[i] == "_":
                origin = i + 1
            if inp[i].isupper():
                break
        origin_end = origin
        for i in range(origin, len(inp)):
            if inp[i] == "_":
                break
            origin_end = i

        poss = []
        for i in range(0, origin + 1):
            for b in range(origin_end + 1, len(inp)):
                possi = inpj[i:b+1]
                if possi.count("_") > len(hand) or possi.count("_") == 0:
                    continue
                if b + 1 != len(inp):
                    if inp[b + 1] != "_":
                        continue
                if i != 0 and inp[i - 1] != "_":
                    break
                poss.append(inp[i:b+1])

        lastp = ""
        for i in range(len(poss)):
            if "".join(poss[i]).replace("_", "") == lastp:
                continue
            lastp = "".join(poss[i]).replace("_", "")
            if "".join(poss[i]).replace("_", "")[-1].lower() == "s":
                essi = poss[i]
                pos = "".join(poss[i]).lower().rfind("s") + 1
                essi = essi[:pos]
                if "".join(essi).count("_") == 0:
                    continue
                essi[-1] = "-"
                poss.append(essi)

        out_dict = []
        for i in range(len(poss)):
            possi = "".join(poss[i]).lower()
            lett = possi.replace("_", "")
            one = possi.find(lett[0])
            two = possi.rfind(lett[-1])
            poss[i] = possi[:one].replace("_", "[a-z]?") + possi[one:two].replace("_", "[a-z]") + possi[two:].replace("_", "[a-z]?")
            if poss[i][-1] == "-":
                r1 = re.compile(poss[i][:-1])
            else:
                r1 = re.compile(poss[i])
            out = list(filter(r1.match, wordset))
            out = list(filter(lambda a: len(lett) <= len(a) <= len(possi), out))
            for h in char_list:
                out = list(filter(lambda a: a.count(h) <= (lett.count(h) + conj[1].count(h)), out))
            if poss[i][-1] == "-":
                out = list(filter(lambda a: p.stem(a + "s") in wordset, out))
                out = [a + "s" for a in out]
                r1 = re.compile(poss[i][:-1] + "s")
                out = list(filter(r1.match, out))
            out = list(filter(lambda a: a != lett, out))
            out_dict = {*out_dict, *out}
        return out_dict

        # first = inp.index("_")
        # for i in range(first, len(inp) - 1):
        #     if inp[i] == "_" and inp[i + 1] == "_":
        #         break_points.append(i)
        # if "".join(inp[-2:]) == "__":
        #     break_points.append(len(inp) - 1)
        #
        # reg = [*["[a-z]" if a=="_" else a for a in inp[:last]], *["[a-z]?" if a=="_" else a for a in inp[last:]]]
        #
        # out_dict = {}
        # for i in break_points:
        #     # print(reg[:i+1])
        #     r1 = re.compile("".join(reg[:i+1]))
        #     lim = i + 1
        #     out = list(filter(r1.match, wordset))
        #     out = list(filter(lambda a: first <= len(a) <= lim, out))
        #     for h in char_list:
        #         out = list(filter(lambda a: a.count(h) <= (conj[0][:i].count(h) + conj[1].count(h)), out))
        #     out_dict = {*out_dict, *out}
        # return out_dict

    def building(xmul=0, ymul=0, xs2=0, ys2=0, base=False, init=""):
        def get_from_board(dist, alt=False):
            if base and dist == 0:
                return init
            xpos = xs2 + (dist * xmul)
            ypos = ys2 + (dist * ymul)
            if 0 > xpos or xpos >= len(board):
                return None
            if 0 > ypos or ypos >= len(board[0]):
                return None
            if alt:
                return [board[xpos][ypos].lower(), (xpos, ypos, xmul, ymul)]
            else:
                return board[xpos][ypos]

        dist = 0
        let = get_from_board(dist)
        leng = len(hand) + 1
        while leng >= 0:
            dist -= 1
            let = get_from_board(dist)
            if let == "_":
                leng -= 1
            if let is None:
                dist += 1
                break
        # while get_from_board(dist - 1) != "_":  # maybe this will work...
        #     dist += 1
        start = dist
        build = []
        build_range = len(hand)
        dist += 1
        while dist < build_range:
            let = get_from_board(dist)
            if let is None:
                break
            if let != "_":
                build_range += 1
            if not base:
                adj = get_from_board(dist, let == "_")
            else:
                adj = get_from_board(dist)
                if adj == "_":
                    break
            adj2 = get_from_board(dist + 1)
            if adj2 is None:
                break
            build.append(adj)
            dist += 1
        return build

    for o in owned:
        xs, ys, letter = o
        build_arr.append(building(1, 0, xs, ys))
        # build_arr.append(building(-1, 0))
        build_arr.append(building(0, 1, xs, ys))
        # build_arr.append(building(0, -1))

    def validate(word, row):  # this does the final test of valid placement
        p = nltk.PorterStemmer()
        arr = []
        for i in range(len(word)):
            if not isinstance(row[i], list):
                continue
            test = building(row[i][1][3], row[i][1][2], row[i][1][0], row[i][1][1], True, word[i])
            if len(test) > 1:
                arr.append("".join(test))
            else:
                continue
        for a in arr:
            if p.stem(a) not in wordset:
                return False
        return True

    build_arr2 = []  # this down here takes data from the board and turns it into possible word lines
    for i in build_arr:
        if i not in build_arr2:
            build_arr2.append(i)
    # build_arr2 = list(map(lambda i: i if i not in build_arr2 else None, build_arr))
    build_arr = build_arr2
    valid_words = []

    for i in range(len(build_arr)):
        row = build_arr[i]
        row_letters = list(map(lambda a: a[0], row))
        # row_letters = [a[0] for a in row]
        if "_" not in row_letters:
            continue
        # print(row_letters)
        words = get_matching(row_letters, debug=True)
        words = list(words)
        if len(words) > 0:
            for w in words:
                if len(w) <= "".join(row_letters).find("_"):
                    continue
                if validate(w, row):
                    valid_words.append(w)

    valid_words = list({*valid_words})

    def score(let):
        arr = [("a", 1), ("b", 3), ("c", 3), ("d", 2), ("e", 1), ("f", 4), ("g", 2), ("h", 4), ("i", 1), ("j", 8), ("k", 5), ("l", 1), ("m", 3), ("n", 1),
               ("o", 1), ("p", 3), ("q", 10), ("r", 1), ("s", 1), ("t", 1), ("u", 1), ("v", 4), ("x", 8), ("y", 4), ("z", 10)]
        score = sum(map(lambda a: let.count(a[0]) * a[1], arr))
        return score

    valid_words.sort()
    valid_words.sort(key=score, reverse=True)
    return valid_words
    # for i in range(math.ceil(len(valid_words) / 10)):
    #     if (i + 1) * 10 < len(valid_words):
    #         print(", ".join(valid_words[i * 10:(i + 1) * 10]))
    #     else:
    #         print(", ".join(valid_words[i * 10:]))


def squareit(inp, img, cube, im_arr):
    char_arr2 = []
    ex, ey, xp, yp = inp
    imc = img.crop((ex, ey, ex + cube, ey + cube))
    im = imc.convert("L")
    w, h = im.size
    area = (round(.24 * w), round(.2 * h), round(.85 * w), round(.8 * h))
    im = im.crop(area)
    tan = check_tan(imc)
    if tan > 0.1:
        im = ImageOps.invert(im)
    im2 = Image.eval(im, (lambda x: (x > 230) * 255))
    hist = im2.histogram()
    perc = hist[255] / (im2.size[0] * im2.size[1])

    char_arr2 = [xp, yp, "_"]

    if perc > 0.07:
        im = normalize(im)

        # im3 = im.copy()
        # im3.save("debug/" + str(yarr.index(ey)) + str(xarr.index(ex)) + ".png")
        teal = check_teal(imc)  # player tiles are teal, ez check is just to see how many teal pixels in square
        num = most_similar2(im, im_arr)
        if teal < 0.2:
            # char_arr2.append(chr(65 + num).lower())
            char_arr2 = [xp, yp, chr(65 + num).lower()]
        else:
            # print(xarr.index(ex), x_ind, yarr.index(ey), y_ind)
            char_arr2 = [xp, yp, chr(65 + num)]
            # char_arr2.append(chr(65 + num))
    return char_arr2


def get_board(xarr, yarr, img):
    pool = ThreadPool(processes=12)
    hand_result = pool.apply_async(get_hand, [img])
    # img = np.asarray(img)
    let_dir = "letters/"
    # letter_files = os.listdir(os.getcwd() + "/" + let_dir)
    # im_arr = [Image.open(let_dir + i).convert("L") for i in letter_files]
    # img2 = img.convert("L")
    # img2 = Image.eval(img2, (lambda x: (x > 240) * 255))  # this is for debug output
    w, h = img.size
    cube = (xarr[-1] - xarr[1]) / (len(xarr) - 1)
    start = (.10 * w, .03 * h)
    end = (.95 * w, .80 * h)
    char_arr = []
    player_squares = []
    letter_files = os.listdir(let_dir)
    # im_arr = [Image.open(let_dir + i).convert("L") for i in letter_files]
    im_arr = list(map(lambda a: Image.open(let_dir + a).convert("L"), letter_files))
    xarr = list(filter(lambda ex: start[0] <= ex <= end[0], xarr))
    yarr = list(filter(lambda ey: start[1] <= ey <= end[1], yarr))

    for x in xarr:
        char_arr2 = []
        for y in yarr:
            char_arr2.append("_")
        char_arr.append(char_arr2)

    for ex in xarr:
        xp = xarr.index(ex)
        for ey in yarr:
            yp = yarr.index(ey)
            imc = img.crop((ex, ey, ex + cube, ey + cube))
            # imc = cv.rectangle(img, (ex, ey), (ex + cube, ey + cube))
            # im = imc.convert("L")
            w, h = imc.size
            area = (round(.24 * w), round(.2 * h), round(.85 * w), round(.8 * h))
            im = imc.convert("L").crop(area)
            # ImageOps.invert(im) if check_tan(imc) > 0.1 else im
            tan = check_tan(imc)
            if tan > 0.1:
                im = ImageOps.invert(im)
            # imt = pool.apply_async(normalize, [im])
            im2 = Image.eval(im, (lambda x: (x > 230) * 255))
            hist = im2.histogram()
            perc = hist[255] / (im2.size[0] * im2.size[1])

            if perc > 0.07:
                # teal = pool.apply_async(check_teal, [imc])
                im = normalize(im)
                # im3 = im.copy()
                # im3.save("debug/" + str(yarr.index(ey)) + str(xarr.index(ex)) + ".png")
                teal = check_teal(imc)  # player tiles are teal, ez check is just to see how many teal pixels in square
                num = most_similar2(im, im_arr)
                # teal = teal.get()
                if teal < 0.2:
                    # char_arr2.append(chr(65 + num).lower())
                    char_arr[xp][yp] = chr(65 + num).lower()
                else:
                    # print(xarr.index(ex), x_ind, yarr.index(ey), y_ind)
                    player_squares.append((xarr.index(ex), yarr.index(ey), chr(65 + num)))
                    char_arr[xp][yp] = chr(65 + num)
                    # char_arr2.append(chr(65 + num))
            else:
                continue
    # print("1.5:", time.time() - startt)
    # startt = time.time()
    # i_arr = []
    # for x in range(len(xarr)):
    #     for y in range(len(yarr)):
    #         i_arr.append([xarr[x], yarr[y], x, y])
    # print(len(i_arr))
    # squar = partial(squareit, img=img, cube=cube, im_arr=im_arr)
    #
    # def add(args):
    #     (a, b, c, d) = args
    #
    # print("1.75:", time.time() - startt)
    # startt = time.time()
    #
    # results = pool.map(squar, i_arr)
    #
    # player_squares = []
    #
    # print("1.9:", time.time() - startt)
    # startt = time.time()
    #
    # for r in results:
    #     x, y, ch = r
    #     char_arr[x][y] = ch
    #     if ch.isupper():
    #         player_squares.append(r)

    # xt = pool.map(x_arr, xarr)

    # print("Played tokens:")
    # print(player_squares)

    # hand_tokens = get_hand(img)
    hand_tokens = hand_result.get()

    # print("Tokens in hand:")
    # print(hand_tokens)

    return hand_tokens, player_squares, char_arr

    # il2 = img2.load()  # for debugging, highlights tiles considered valid during processing
    # for i in fin_arr:
    #     for x in range(i[0], i[0] + round(cube)):
    #         for y in range(i[1], i[1] + round(cube)):
    #             a = il2[x, y]
    #             if a < 200:
    #                 a = 150
    #             il2[x, y] = a
    # img2.show()  # for showing the above result


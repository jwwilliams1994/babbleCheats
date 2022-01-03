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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import nltk
# from nltk.corpus import words as wd
# from nltk.corpus import wordnet
import matplotlib.pyplot as plt
# import inflect
import json

let_dir = "letters/"
letter_files = os.listdir(os.getcwd() + "/" + let_dir)
# pe = inflect.engine()
# wordset1 = set(wd.words())
# wordset = list({*[*wordset1, *map(lambda a: pe.plural(a), wordset1.copy())]})

# json_obj = json.dumps(wordset, indent=4)
filename = "extra_wordset.json"
with open(filename) as json_file:
    wordset = json.load(json_file)
# with open(filename, 'w') as f:
#     f.write(json_obj)


def dist(rgb, col=(118, 103, 80), v1=0.1, v2=0.15, v3=0.15):  # default col is average grid color
    h, s, v = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    if abs(h - col[0]) < v1:
        if abs(s - col[1]) < v2:
            if abs(v - col[2]) < v3:
                return True
    return False


def dist2(rgb, col=(118, 103, 80), v1=26, v2=38, v3=38):  # default col is average grid color
    # h, s, v = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    h, s, v = rgb
    if abs(h - col[0]) < v1:
        if abs(s - col[1]) < v2:
            if abs(v - col[2]) < v3:
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
    # arr = list(map(lambda a: a - offset, inp))[1:]
    arr = [a - offset for a in inp][1:]
    m_arr = []
    # for i in range(1, len(arr)):
    #     m_arr.append(arr[i] - arr[i - 1])
    m_arr = list(map(lambda i: arr[i] - arr[i - 1], range(1, len(arr))))

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


def plotty(xarr, yarr, lim):
    fig = plt.figure()
    host = fig.add_subplot()
    host.plot(xarr)
    fig2 = plt.figure()
    part = fig2.add_subplot()
    part.plot(yarr)
    plt.show()


def get_grid(inp, val=1):
    w, h = inp.size
    inp = inp.resize((round(w / val), round(h / val)), Image.NEAREST)
    w, h = inp.size
    img1 = inp.convert("HSV").copy()
    img2 = inp.convert("L").copy()

    iml = img1.load()
    il2 = img2.load()
    col = (118, 103, 80)
    col = colorsys.rgb_to_hsv(col[0] / 255, col[1] / 255, col[2] / 255)
    for x in range(w):
        for y in range(h):
            rgb = iml[x, y]
            if dist(rgb, col):
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


def get_similarity(img, img2):
    img2 = np.asarray(img2)
    out = round(mean_squared_error(img, img2), 3)
    return out


def get_sim(img, img2, debug=False):  # this one is currently used, above and below are ignored
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
    if "H" in letters and "U" in letters:
        img = img.crop((5, 8, 20, 16))
    return img


def most_similar2(im, im_list, diff=False, debug=False):
    let_arr = ["Q", "O", "G", "E", "F", "L", "B", "D", "R", "P", "U", "H"]
    ima = np.asarray(im)
    if diff:
        arr = list(map(lambda a: get_sim(im, a, debug), im_list))
        # arr = [get_sim(im, a) for a in im_list]
        arr.sort()
        return arr[0]
    sim_arr = []
    o_arr = [*zip(map(lambda a: get_sim(ima, a, debug), im_list), list(string.ascii_uppercase))]
    o_arr.sort()
    letters = [o_arr[0][1], o_arr[1][1]]  # to make doubly sure it's what it's supposed to be, the matching can be otherwise drunk with similar characters
    if letters[0] in let_arr or letters[1] in let_arr:
        o_arr = [*zip(map(lambda a: (get_sim(np.asarray(sieve(im, letters)), sieve(a, letters))), im_list), list(string.ascii_uppercase))]
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
    col = colorsys.rgb_to_hsv(col[0] / 255, col[1] / 255, col[2] / 255)
    col = [int(a * 255) for a in col]
    # col = list(map(lambda a: int(a * 255), col))
    count = sum(map(lambda i: dist2(i, col), im_iter))
    perc = count / (img.size[0] * img.size[1])
    return perc


def check_tan(img):
    col = (255, 246, 167)
    im_iter = img.getdata()
    col = colorsys.rgb_to_hsv(col[0] / 255, col[1] / 255, col[2] / 255)
    # col = list(map(lambda a: int(a * 255), col))
    col = [int(a * 255) for a in col]
    count = sum(map(lambda i: dist2(i, col), im_iter))
    perc = count / (img.size[0] * img.size[1])
    return perc


def check_hand_color(img):
    col = (77, 111, 111)
    im_iter = img.convert("HSV").getdata()
    col = colorsys.rgb_to_hsv(col[0] / 255, col[1] / 255, col[2] / 255)
    col = list(map(lambda a: int(a * 255), col))
    count = sum(map(lambda i: dist2(i, col), im_iter))
    perc = count / (img.size[0] * img.size[1])
    return perc


def get_letter(img, debug=False):  # img needs to be normalized first
    im_arr = list(map(lambda i: Image.open(let_dir + i).convert("L"), letter_files))
    # im_arr = [Image.open(let_dir + i).convert("L") for i in letter_files]
    num = most_similar2(img, im_arr, debug)
    return chr(num + 65)


def get_hand(img, debug=False):
    w, h = img.size
    il = img.convert("HSV").load()
    col = (76, 111, 111)
    col = colorsys.rgb_to_hsv(col[0] / 255, col[1] / 255, col[2] / 255)
    x = round(w / 2)
    st = 0
    en = 0
    for y in range(round(h * 0.8), h):
        rgb = il[x, y]
        if dist(rgb, col, .02, .05, .08):
            if st == 0:
                st = y
            en = y
    yrat1 = (st / h)
    yrat2 = (en / h)
    yratd = yrat2 - yrat1
    yrat1 = yrat1 + (yratd * 0.3)
    yrat2 = yrat2 - (yratd * 0.3)
    # print(st, en)
    midy = round(h * yrat1 + 5)
    st = 0
    en = 0
    for x in range(round(w * 0.1), round(w * 0.9)):
        rgb = il[x, midy]
        if dist(rgb, col, .02, .05, .08):
            if st == 0:
                st = x
            en = x
    xrat1 = (st / w)
    xrat2 = (en / w)
    xratd = xrat2 - xrat1
    xrat1 = xrat1 + (xratd * 0.015)
    xrat2 = xrat2 - (xratd * 0.013)
    # print(st, en)
    start = (round(xrat1 * w), round(yrat1 * h))
    end = (round(xrat2 * w), round(yrat2 * h))
    img = img.crop((*start, *end)).convert("L")  # crops out the area the hand tiles exist in, no reason to bother looking elsewhere
    if debug:
        img.save('debug/hand.png')
    # img = Image.eval(img, (lambda x: (x < 20) * 255))
    # img.show()
    w, h = img.size
    # port = w / 10
    hand_arr = []
    left = None
    right = w - 1
    top = None
    bot = h - 1
    il = img.load()
    midy = round(h / 2)
    count = 0
    count2 = 0
    count3 = 0
    st_arr = []
    wid_arr = []
    for x in range(w):
        if il[x, midy] > 230:
            if left is None:
                if top is None:
                    midy = bot
                    for y in range(h):
                        if il[x, y] > 230:
                            if top is None:
                                top = y
                        if il[x, y] < 230:
                            if top is not None:
                                if count2 < 2:
                                    bot = y - 2
                                else:
                                    midy = bot
                                    break
                left = x
        if il[x, midy] <= 50:
            continue
        if left is not None:
            if 50 < il[x, midy] < 220:
                count += 1
            if count < 5:
                right = x - 5
            else:
                count = 0
                st_arr.append([left, right])
                left = None
                right = w

    for st in st_arr:
            left, right = st
            width = (right - left) * 0.12
            area = (round(left + width), round(top), round(right - width), round(bot))
            out = img.crop(area)
            out = Image.eval(out, (lambda x: (x < 20) * 255))
            if debug:
                out.save(str(count3) + ".png")
                count3 += 1
            hist = out.histogram()[255] / (out.size[0] * out.size[1])  # percentage of pixels that are full white
            if hist > 0.02:
                out = get_letter(normalize(out))
                hand_arr.append(out)
    return hand_arr

    # for i in range(10):
    #     area = (round(port * i), 0, round(port * (i + 1)), h)
    #     out = img.crop(area)
    #     w, h = out.size
    #     area = (round(.14 * w), round(.00 * h), round(.86 * w), round(.9 * h))
    #     out = out.crop(area)
    #     if debug:
    #         out.save(str(i) + ".png")
    #     hist = out.histogram()[255] / (out.size[0] * out.size[1])  # percentage of pixels that are full white
    #     if hist > 0.02:
    #         out = get_letter(normalize(out), True)
    #         hand_arr.append(out)
    # return hand_arr


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
        for i in range(0, origin+1):
            for b in range(origin_end, len(inp)):
                possi = inpj[i:b+1]
                # print(possi)
                if possi.count("_") > len(hand) or possi.count("_") == 0:
                    # print("fail 1")
                    continue
                if b + 1 != len(inp):
                    if inp[b + 1] != "_":
                        # print("fail 2")
                        continue
                if i != 0 and inp[i - 1] != "_":
                    # print("fail 3")
                    break
                poss.append(inp[i:b+1])

        # lastp = ""
        # for i in range(len(poss)):
        #     if "".join(poss[i]).replace("_", "") == lastp:
        #         continue
        #     lastp = "".join(poss[i]).replace("_", "")
        #     if "".join(poss[i]).replace("_", "")[-1].lower() == "s":
        #         essi = poss[i]
        #         pos = "".join(poss[i]).lower().rfind("s") + 1
        #         essi = essi[:pos]
        #         if "".join(essi).count("_") == 0:
        #             continue
        #         essi[-1] = "-"
        #         poss.append(essi)

        # print(poss[0], poss[-1])
        out_list = []
        out_dict = []

        t_arr = []
        arr1 = poss[0].copy()
        arr2 = poss[-1].copy()
        ind = arr2.index(arr1[-1]) + 1

        base_tiles = "".join(list(filter(lambda a: a.isupper(), arr1.copy() + arr2[ind:].copy()))).lower()
        base_word = base_tiles.lower()
        lower_limit = sum(map(lambda a: a.isupper(), arr1))

        handj = "".join([*{*hand}])
        lett = ("".join(arr1) + "".join(arr2)).replace("_", "").lower()
        this_char_list = {*lett, *handj}
        original_string = "".join(arr1 + arr2[ind:])
        part = "[" + handj + "]"
        count = 0
        for i in range(1, len(arr1) + 1):
            if arr1[-i] == "_":
                count += 1
                arr1[-i] = part + ")?"
        while count > 0:
            count -= 1
            arr1[0] = "(" + arr1[0]

        es = False
        count = 0
        for i in range(1, len(arr2)):
            if arr2[i].lower() == "s" and i < len(arr2) - 1:
                if arr2[i + 1] == "_":
                    es = True
                    arr2[i] = "(s"
                    count += 1
            if arr2[i] == "_":
                count += 1
                arr2[i] = "(" + part
        while count > 0:
            count -= 1
            arr2[-1] = arr2[-1] + ")?"

        arr3 = "\\b" + "".join(arr1) + "".join(arr2[ind:]) + "\\b"

        r1 = re.compile(arr3.lower())
        out = wordset
        out = list(filter(r1.match, out))
        reg = "^.{" + str(lower_limit + 1) + ",}$"
        r2 = re.compile(reg)
        out = list(filter(r2.match, out))
        # out = list(filter(lambda a: len(a) > lower_limit, out))
        for h in this_char_list:
            out = list(filter(lambda a: a.count(h) <= (base_word.count(h) + handj.count(h)), out))
        # out = list(filter(lambda a: p.stem(a) in wordset1, out))

        check_arr = list(filter(lambda a: original_string[a] != "_", range(len(original_string))))
        check_arr2 = list(filter(lambda a: original_string[a].isupper(), range(len(original_string))))
        original_string2 = original_string.lower()

        def cap_origin(inp):  # the logic isn't _100%_ airtight, but it should work basically every time
            for i in range(len(inp)):
                if inp[i] in base_tiles:
                    ind = check_arr.index(original_string.find(inp[i].upper()))
                    offset = check_arr[ind] - i
                    check_set = list(filter(lambda a: 0 <= (a - offset) < len(inp), check_arr))
                    correct = list(map(lambda a: original_string2[a] == inp[a - offset], check_set))
                    # correct = [(original_string2[a] == inp[a - offset]) for a in check_set]
                    if False in correct:
                        continue
                    else:
                        output = "".join(map(lambda a: inp[a].upper() if a + offset in check_set else inp[a], range(len(inp))))
                        return output
            return inp

        out_dict = {*out}
        if len(out) > 0:
            test_arr = list(map(lambda a: cap_origin(a), out_dict))
            out_dict = {*test_arr}

        return out_dict

    def building(xmul=0, ymul=0, xs2=0, ys2=0, base=False, init="", debug=False):
        def get_from_board(bdist, alt=False):
            if base and bdist == 0:
                return init
            xpos = xs2 + (bdist * xmul)
            ypos = ys2 + (bdist * ymul)
            if 0 > xpos or xpos >= len(board):
                return None
            if 0 > ypos or ypos >= len(board[0]):
                return None
            if alt:
                return [board[xpos][ypos].lower(), (xpos, ypos, xmul, ymul)]
            else:
                return board[xpos][ypos]
        bdist = 0
        if not base:
            let = get_from_board(bdist)
            leng = len(hand) + 1
            while leng >= 0:
                bdist -= 1
                let = get_from_board(bdist)
                if let == "_":
                    leng -= 1
                if let is None:
                    bdist += 1
                    break
        else:
            let = get_from_board(bdist)
            while let != "_" and let is not None:
                bdist -= 1
                let = get_from_board(bdist)
            if let == "_":
                bdist += 1
        # while get_from_board(dist - 1) != "_":  # maybe this will work...
        #     dist += 1
        start = bdist
        build = []
        build_range = len(hand)
        if base:
            build_range = 10
        else:
            bdist += 1
        while bdist < build_range:
            let = get_from_board(bdist)
            if let is None:
                break
            if let != "_":
                build_range += 1
            if not base:
                bol = let=="_"
                adj = get_from_board(bdist, bol)
            else:
                adj = get_from_board(bdist)
                if adj == "_":
                    break
            adj2 = get_from_board(bdist + 1)
            if adj2 is None:
                break
            build.append(adj)
            bdist += 1
        if debug:
            print(build)
        return build

    for o in owned:
        xs, ys, letter = o
        build_arr.append(building(1, 0, xs, ys))
        # build_arr.append(building(-1, 0))
        build_arr.append(building(0, 1, xs, ys))
        # build_arr.append(building(0, -1))

    def validate(word, row, debug=False):  # this does the final test of valid placement
        row_string = "".join([a[0] for a in row]).lower()
        match_word = "".join(map(lambda a: a if a.isupper() else "_", word)).lower()
        ind = row_string.find(match_word)
        p = nltk.PorterStemmer()
        arr = []
        for i in range(ind, len(word) + ind):
            if not isinstance(row[i], list):  # if not list...
                continue
            test = building(row[i][1][3], row[i][1][2], row[i][1][0], row[i][1][1], True, word[i - ind])
            if len(test) > 1:
                arr.append("".join(test))
            else:
                continue
        for a in arr:
            if p.stem(a.lower()) not in wordset:
                return False
        return True

    build_arr2 = []  # this down here takes data from the board and turns it into possible word lines
    for i in build_arr:
        if i not in build_arr2:
            build_arr2.append(i)
    # build_arr2 = list(map(lambda i: i if i not in build_arr2 else None, build_arr))
    build_arr = build_arr2


    moves_arr = []
    counter = 0
    for i in range(len(build_arr)):
        counter += 1
        valid_words = []
        row = build_arr[i]
        row_letters = [a[0] for a in row]
        if "_" not in row_letters:
            continue
        # print(row_letters)
        words = get_matching(row_letters)
        words = list(words)
        if len(words) > 0:
            for w in words:
                if len(w) <= "".join(row_letters).find("_"):
                    continue
                if validate(w, row):
                    valid_words.append(w)
            moves_arr.append(list({*valid_words}))


    # valid_words = list({*valid_words})

    def score(let):
        arr = [("a", 1), ("b", 3), ("c", 3), ("d", 2), ("e", 1), ("f", 4), ("g", 2), ("h", 4), ("i", 1), ("j", 8), ("k", 5), ("l", 1), ("m", 3), ("n", 1),
               ("o", 1), ("p", 3), ("q", 10), ("r", 1), ("s", 1), ("t", 1), ("u", 1), ("v", 4), ("x", 8), ("y", 4), ("z", 10)]
        score = sum(map(lambda a: let.count(a[0]) * a[1], arr))
        return score

    for i in range(len(moves_arr)):
        valid_words = moves_arr[i]
        valid_words.sort()
        valid_words.sort(key=score, reverse=True)
        moves_arr[i] = valid_words

    return moves_arr
    # for i in range(math.ceil(len(valid_words) / 10)):
    #     if (i + 1) * 10 < len(valid_words):
    #         print(", ".join(valid_words[i * 10:(i + 1) * 10]))
    #     else:
    #         print(", ".join(valid_words[i * 10:]))


def get_board(xarr, yarr, img):
    let_dir = "letters/"
    w, h = img.size
    cube = round(xarr[-1] - xarr[1]) / (len(xarr) - 1)
    start = (.10 * w, .03 * h)
    end = (.95 * w, .80 * h)
    char_arr = []
    player_squares = []
    letter_files = os.listdir(let_dir)
    im_arr = list(map(lambda a: Image.open(let_dir + a).convert("L"), letter_files))
    xarr = list(filter(lambda ex: start[0] <= ex <= end[0], xarr))
    yarr = list(filter(lambda ey: start[1] <= ey <= end[1], yarr))

    for x in xarr:
        char_arr2 = []
        for y in yarr:
            char_arr2.append("_")
        char_arr.append(char_arr2)

    img_squares = []

    img = img.convert("HSV")

    for ex in xarr:
        xp = xarr.index(ex)
        for ey in yarr:
            yp = yarr.index(ey)
            imc = img.crop((ex, ey, ex + cube, ey + cube))
            w, h = imc.size
            area = (int(.24 * w), int(.2 * h), int(.85 * w), int(.8 * h))
            im = imc.convert("L").crop(area)

            tan = check_tan(imc)
            if tan > 0.1:
                im = ImageOps.invert(im)

            im2 = Image.eval(im, (lambda x: (x > 230) * 255))
            hist = im2.histogram()
            perc = hist[255] / (im2.size[0] * im2.size[1])

            if perc > 0.07:

                im = normalize(im)

                teal = check_teal(imc)  # player tiles are teal, ez check is just to see how many teal pixels in square

                num = most_similar2(im, im_arr)

                if teal < 0.2:
                    # char_arr2.append(chr(65 + num).lower())
                    char_arr[xp][yp] = chr(65 + num).lower()
                else:
                    # print(xarr.index(ex), x_ind, yarr.index(ey), y_ind)
                    player_squares.append((xp, yp, chr(65 + num)))
                    char_arr[xp][yp] = chr(65 + num)
                    # char_arr2.append(chr(65 + num))
            else:
                continue

    return player_squares, char_arr


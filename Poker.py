import cv2
import numpy as np
import math
import argparse
from os import walk


def GoodPointsTrain(img, bf):
    matches = bf.knnMatch(img, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def GoodPoints(img, img2, bf):
    matches = bf.knnMatch(img, img2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good.append(m)
    return good


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def Draw(images, tablePath, name, withContour):
    table = cv2.imread(tablePath)
    img_lst = list()
    img_color = list()
    orb_img = cv2.ORB_create(nfeatures=3000, edgeThreshold=1)
    orb_table = cv2.ORB_create(nfeatures=25000, edgeThreshold=1)
    bf = cv2.BFMatcher()
    if images is not None:
        for image in images:
            img = cv2.imread(image)
            img_lst.append(img)
            img_color.append(list(np.random.choice(range(256), size=3)))

    des_list = list()
    kp_lst = list()
    for image in img_lst:
        kp, des = orb_table.detectAndCompute(image, None)
        des_list.append(des)
        kp_lst.append(kp)
    if withContour:
        bf.add(np.array(des_list, dtype=object))
        bf.train()

        gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_copy = table.copy()
        cleancopy = table.copy()
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        image_copy = cv2.resize(image_copy, (int(image_copy.shape[1] / 2), int(image_copy.shape[0] / 2)))
        cv2.imshow("table", image_copy)
        cv2.waitKey(0)
        cardAlreadyPicked = list()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (500 > h > 350 and 400 > w > 150) or (500 > w > 350 and 400 > h > 150):
                carte = cleancopy[y:y + h + 20, x:x + w + 20]
                kp, des = orb_img.detectAndCompute(carte, None)
                maxIndex = -1
                good = GoodPointsTrain(des, bf)
                arrayIdx = [m.imgIdx for m in good]
                if len(arrayIdx) > 0:
                    maxIndex = np.bincount(arrayIdx).argmax()
                    while maxIndex in cardAlreadyPicked:
                        arrayIdx.remove(maxIndex)
                        maxIndex = np.bincount(arrayIdx).argmax()
                maxGood = len([m for m in good if m.imgIdx == maxIndex])

                cardAlreadyPicked.append(maxIndex)
                print(name[maxIndex] + " GoodPoints: " + str(maxGood))

                cv2.rectangle(table, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(table, (name[maxIndex].format(w, h)),
                            (x + 30, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            3,
                            cv2.LINE_AA)
        table = cv2.resize(table, (int(table.shape[1] / 2), int(table.shape[0] / 2)))
        cv2.imshow("table", table)
        cv2.waitKey(0)
    else:
        i = -1
        kp_table, des_table = orb_table.detectAndCompute(table, None)
        for img in img_lst:
            i = i + 1
            kp_img, des_img = orb_table.detectAndCompute(img, None)
            good = GoodPoints(des_img, des_table, bf)
            print(name[i] + ": " + str(len(good)))

            if len(good) > 40:
                sch_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                img_pts = np.float32([kp_table[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                matrix, mask = cv2.findHomography(sch_pts, img_pts, cv2.RANSAC, 2.0)
                if matrix is None:
                    print("Matrix None")
                if matrix is not None:
                    h, w, t = img.shape
                    pts = np.float32([[0, 0],
                                      [0, h - 1],
                                      [w - 1, h - 1],
                                      [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, matrix)
                    lstError = list()
                    for errorIdx in range(4):
                        point = dst[errorIdx][0]
                        if point[0] < 0 or point[1] < 0 or point[0] > table.shape[1] or point[1] > table.shape[0]:
                            lstError.append(point)
                    if len(lstError) >= 0:
                        col = img_color[i]

                        font = cv2.FONT_ITALIC
                        bottomLeftCornerOfText = (int(dst[0][0][0]), int(dst[0][0][1]))
                        fontScale = 5
                        fontColor = (0, 0, 255)
                        thickness = 3
                        lineType = 2
                        angle_bas = getAngle(dst[0][0], dst[1][0], dst[2][0])
                        angle_haut = getAngle(pts[2][0], pts[3][0], pts[0][0])
                        if (70 < angle_bas < 110) and (70 < angle_haut < 110):
                            table = cv2.putText(table, name[i],
                                                bottomLeftCornerOfText,
                                                1,
                                                fontScale,
                                                fontColor,
                                                thickness,
                                                lineType)
                            table = cv2.polylines(table, [np.int32(dst)], True, [int(col[0]), int(col[1]), int(col[2])],
                                                  3,
                                                  cv2.LINE_AA)
                        else:
                            print("bad angle")
                    else:
                        print("Error found:" + str(len(lstError)))

        table = cv2.resize(table, (int(table.shape[1] / 2), int(table.shape[0] / 2)))
        table = cv2.resize(table, (int(table.shape[1] / 2), int(table.shape[0] / 2)))

        cv2.imshow("table", table)
        cv2.waitKey(0)


withContour = True
Images = list()
name = list()
monRepertoire = "carte_from_table/"
tablePath = "tables_de_poker/Table2.jpg"
for (repertoire, sousRepertoires, fichiers) in walk(monRepertoire):
    for nameFile in fichiers:
        if nameFile.split('.')[1] == "jpg":
            pathImg = monRepertoire + nameFile
            name.append(nameFile.split('.')[0])
            Images.append(pathImg)

Draw(Images, tablePath, name, withContour)

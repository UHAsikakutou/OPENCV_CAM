import cv2
def compare(imported):
    # 切り取った顔領域の大きさを調整する時に使う値
    IMG_SIZE = (200, 200)

    # カスケード機生成
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')   
    # 画像ファイルの読み込み
    img1 = cv2.imread(imported)
    img2 = cv2.imread("sample1.jpg")
 
    # グレースケール変換
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
# 顔領域取得
    img1_faces = face_cascade.detectMultiScale(img1_gray, minSize=(100, 100))
    img2_faces = face_cascade.detectMultiScale(img2_gray, minSize=(100, 100))
 
# 画の顔領域の画像の座標を取得
    img1_face_rect = img1_faces[0]
    img2_face_rect = img2_faces[0]
    x1, y1, w1, h1 = img1_face_rect[0],img1_face_rect[1],img1_face_rect[2],img1_face_rect[3]
    x2, y2, w2, h2 = img2_face_rect[0],img2_face_rect[1],img2_face_rect[2],img2_face_rect[3]
 
    # 画像から顔領域を抽出し大きさを200x200に変更
    img1_face = img1[y1:y1+h1, x1:x1+w1]
    img2_face = img2[y2:y2+h2, x2:x2+w2]
    img1_face = cv2.resize(img1_face, IMG_SIZE)
    img2_face = cv2.resize(img2_face, IMG_SIZE)
 
    # AKAZE特徴量を使った特徴点検出の準備
    akaze = cv2.AKAZE_create()
 
    # 二つの顔領域の特徴点を取得
    (img1_face_kp, img1_face_des) = akaze.detectAndCompute(img1_face, None)
    (img2_face_kp, img2_face_des) = akaze.detectAndCompute(img2_face, None)
 
    # BFMatcherを定義
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # BFMatcherで総当たりマッチングを行う
    matches = bf.match(img1_face_des, img2_face_des)
 
    #特徴量の距離を出し、平均を取る
    dist = [m.distance for m in matches]
    if len(dist) != 0:
        ret = sum(dist) / len(dist)
    comparing_img = cv2.drawMatches(img1_face, img1_face_kp, img2_face, img2_face_kp, matches[:10], None, flags=2)
    cv2.imwrite("compare.jpg", comparing_img)
    return ret
import cv2
img=cv2.imread("output2.png")
cv2.imshow('My Image', img)
print(type(img))
# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

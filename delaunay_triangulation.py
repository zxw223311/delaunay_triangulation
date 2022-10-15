import cv2
import dlib

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


if __name__ == '__main__':

    # Define window names
    delaunay = "Delaunay Triangulation"

    # Turn on animation while drawing triangles
    animate = True

    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 255, 0)

    # Read in the image.
    img = cv2.imread('1.jpg')

    # Keep a copy around
    img_orig = img.copy()

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)
    # Create an array of points.
    points = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 检测器检测人脸
    faces = detector(img_gray, 0)
    if (len(faces) != 0):
        # 对每个人脸都标出68个特征点
        # enumerate 方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
        for k, d in enumerate(faces):
            #  rectangle(img, pt1, pt2, color), 其中pt1为矩阵上顶点，pt2为矩阵下顶点
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
            # 使用预测器得到68点数据的坐标
            shape = predictor(img, d)
            # 圆圈显示每个特征点
            for i in range(68):
                subdiv.insert((shape.part(i).x, shape.part(i).y))
                points.append([shape.part(i).x, shape.part(i).y])
                # Show animation
                if animate:
                    img_copy = img_orig.copy()
                    # Draw delaunay triangles
                    draw_delaunay(img_copy, subdiv, (255, 255, 255))
                    cv2.imshow(delaunay, img_copy)
                    cv2.waitKey(100)

            # Draw delaunay triangles
            draw_delaunay(img, subdiv, (255, 255, 255))
            # Draw points
            for p in points:
                draw_point(img, p, (255, 0, 0))

            cv2.imshow(delaunay, img)
            cv2.waitKey(0)


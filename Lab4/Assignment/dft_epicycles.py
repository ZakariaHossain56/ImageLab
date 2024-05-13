
from collections import deque
from canny_edge_detection import *
from animator import start_animation



def dfs(image, sp_x, sp_y, to_replace, replace_with):
    height, width = image.shape
    parent_map = {}
    length = 0
    last = None

    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None

    while stack:
        x, y, it = stack.pop()
        if image[x, y] != to_replace:
            continue

        image[x, y] = replace_with

        it += 1
        if it > length:
            length = it
            last = (x, y)

        indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        for dx, dy in indices:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and image[nx, ny] == to_replace:
                if (nx, ny) not in parent_map:  # Check if not already visited
                    parent_map[(nx, ny)] = (x, y)
                    stack.append((nx, ny, it))

    points = []
    while last is not None:
        points.append(last)
        last = parent_map[last]
    points.reverse()
    return points


def get_edge_points(image):
    image = image.copy()
    height, width = image.shape
    pad = 10
    image = image[pad:height - pad, pad:width - pad]
    height, width = image.shape
    contours = []

    visited = {}

    def bfs(sx, sy):
        nonlocal visited
        to_it = (sx, sy)

        while to_it != None:
            queue = deque()
            queue.append(to_it)

            to_it = None
            count = 0
            while queue:
                x, y = queue.popleft()
                if visited.get((x, y)) == True:
                    continue
                count += 1

                image[x, y] = 60

                indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                for dx, dy in indices:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= height or ny < 0 or ny >= width or image[
                        nx, ny] == 60:
                        continue

                    if image[nx, ny] == 255:
                        to_it = (nx, ny)
                        queue.clear()
                        break
                    if visited.get(to_it) == None:
                        queue.append((nx, ny))
                visited[(x, y)] = True

            if to_it == None:
                break

            points = dfs(image, to_it[0], to_it[1], to_replace=255, replace_with=120)
            last_pt = points[len(points) - 1]

            points = dfs(image, last_pt[0], last_pt[1], to_replace=120, replace_with=60)
            if (len(points) > 20):
                contours.append(points)

            to_it = points[len(points) - 1]

    for x in range(height):
        for y in range(width):
            if visited.get((x, y)) == None:
                bfs(x, y)
    cv2.imshow("After contours", image)

# 2D contours to 1D
    points = []
    for i in range(len(contours)):
        cnt = contours[i]
        for pt in cnt:
            points.append((pt[0] / 2, pt[1] / 2))
    return points




imagepath = "penguin.jpg"
image = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input grayscale image",image)
cv2.waitKey(0)
edges = canny(imagepath)
cv2.imshow("Canny edge detection",edges)
cv2.waitKey(0)

edge_points = get_edge_points(edges)

# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# edge_points = []
# for contour in contours:
#     for point in contour:
#         edge_points.append(tuple(point[0]))


start_animation(edge_points)




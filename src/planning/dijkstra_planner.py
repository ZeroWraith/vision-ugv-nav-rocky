import heapq
import numpy as np


def _heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def dijkstra(costmap, origin, start_xy, goal_xy, resolution=0.05):
    """
    Dijkstra on costmap with cost-aware traversal.

    costmap : 2D uint8 (0 free, 254 occupied, 127 unknown, gradient in between)
    origin  : world xy of grid[0,0]
    Returns list of world (x,y) waypoints from start to goal.
    """
    H, W = costmap.shape

    def world_to_grid(pt):
        gx = int((pt[0] - origin[0]) / resolution)
        gy = int((pt[1] - origin[1]) / resolution)
        return (gx, gy)

    def grid_to_world(g):
        return (g[0] * resolution + origin[0], g[1] * resolution + origin[1])

    start = world_to_grid(start_xy)
    goal = world_to_grid(goal_xy)

    start = (max(0, min(W - 1, start[0])), max(0, min(H - 1, start[1])))
    goal = (max(0, min(W - 1, goal[0])), max(0, min(H - 1, goal[1])))

    if costmap[start[1], start[0]] >= 250 or costmap[goal[1], goal[0]] >= 250:
        return [start_xy, goal_xy]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (-1, -1), (1, -1), (-1, 1)]
    move_cost = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return [grid_to_world(p) for p in path]

        for (dx, dy), mc in zip(moves, move_cost):
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            cell_cost = costmap[ny, nx]
            if cell_cost >= 250:
                continue
            # Cost-weighted: prefer low-cost terrain
            edge_weight = mc * (1.0 + cell_cost / 254.0)
            tentative = g_score[cur] + edge_weight
            nxt = (nx, ny)
            if tentative < g_score.get(nxt, 1e9):
                came_from[nxt] = cur
                g_score[nxt] = tentative
                f = tentative  # Dijkstra: no heuristic
                heapq.heappush(open_set, (f, nxt))

    return [start_xy, goal_xy]
import heapq
import numpy as np

def _heuristic(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def astar(costmap, origin, start_xy, goal_xy, resolution=0.05):
    """
    costmap : 2D uint8 (0 free, 255 occupied, 127 unknown)
    origin  : world xy of grid[0,0]
    Returns list of world (x,y) waypoints from start to goal.
    """
    H, W = costmap.shape
    def world_to_grid(pt):
        gx = int((pt[0] - origin[0]) / resolution)
        gy = int((pt[1] - origin[1]) / resolution)
        return (gx, gy)

    def grid_to_world(g):
        return (g[0]*resolution + origin[0], g[1]*resolution + origin[1])

    start = world_to_grid(start_xy)
    goal  = world_to_grid(goal_xy)

    # clamp
    start = (max(0,min(W-1,start[0])), max(0,min(H-1,start[1])))
    goal  = (max(0,min(W-1,goal[0])),  max(0,min(H-1,goal[1])))

    if costmap[start[1], start[0]] == 255 or costmap[goal[1], goal[0]] == 255:
        # fallback: return straight line
        return [start_xy, goal_xy]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: _heuristic(start, goal)}

    moves = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
    move_cost = [1,1,1,1,1.414,1.414,1.414,1.414]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            # reconstruct
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return [grid_to_world(p) for p in path]

        for (dx,dy), mc in zip(moves, move_cost):
            nx, ny = cur[0]+dx, cur[1]+dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if costmap[ny, nx] == 255:
                continue
            tentative = g_score[cur] + mc
            nxt = (nx, ny)
            if tentative < g_score.get(nxt, 1e9):
                came_from[nxt] = cur
                g_score[nxt] = tentative
                f = tentative + _heuristic(nxt, goal)
                f_score[nxt] = f
                heapq.heappush(open_set, (f, nxt))

    # no path found
    return [start_xy, goal_xy]
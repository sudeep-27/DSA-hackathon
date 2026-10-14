"""Lockstep: shortest paths through a maze of locked doors.

One-file web app for the DSA hackathon problem "Maze with Keys and Doors".
Backend: Flask + the algorithms. Frontend: the page embedded at the bottom.

Run:  python -m pip install flask
      python maze_app.py
Open: http://127.0.0.1:5000

Grid symbols (CSV, N x N):
  .  open floor        #  wall
  S  start             E  exit
  a-f keys             A-F doors (door X opens only with key x)
"""
import csv
import heapq
import io
import random
import time
from collections import deque

from flask import Flask, Response, jsonify, request

# ============================================================== algorithms
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
KEYS = "abcdef"
DOORS = "ABCDF"          # "E" is the exit, so a door E can never exist
GEN_KEYS = "abcdf"       # keys the generator uses (skips e for the same reason)
EXPLORE_CAP = 120_000    # most states sent to the browser for the animation


def validate_grid(grid):
    if not isinstance(grid, list) or not grid or any(not isinstance(r, list) or not r for r in grid):
        raise ValueError("The grid is empty.")
    if any(len(row) != len(grid) for row in grid):
        raise ValueError("The maze must be an N x N square grid.")
    if len(grid) > 81:
        raise ValueError("Keep the maze at 81 x 81 or smaller.")
    allowed = set(".#SEabcdefABCDEF")
    bad = {ch for row in grid for ch in row if ch not in allowed}
    if bad:
        raise ValueError(f"Unknown cell {sorted(bad)[0]!r}. Use . # S E a-f A-F.")
    flat = [ch for row in grid for ch in row]
    if flat.count("S") != 1:
        raise ValueError("Place exactly one start (S).")
    if flat.count("E") != 1:
        raise ValueError("Place exactly one exit (E).")
    return grid


def find(grid, target):
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == target:
                return r, c
    return None


def key_bit(ch):
    return 1 << (ord(ch.lower()) - ord("a"))


def neighbors(grid, state):
    """Legal moves from (row, col, mask). Picking up a key sets its bit."""
    r, c, mask = state
    n = len(grid)
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n and 0 <= nc < n):
            continue
        ch = grid[nr][nc]
        if ch == "#":
            continue
        m = mask
        if ch in DOORS:
            if not mask & key_bit(ch):
                continue                     # locked: we do not hold its key
        elif ch in KEYS:
            m |= key_bit(ch)
        yield nr, nc, m


def solve(grid, algo="bfs"):
    """Shortest path from S to E.

    bfs   : BFS over states (row, col, keys-held bitmask). Optimal.
    astar : A* over the same states, Manhattan distance to E as heuristic.
            The heuristic never overestimates, so the answer is still optimal.
    naive : BFS that remembers only (row, col). This is the classic mistake:
            once a cell is visited it is never re-entered, even after a key
            makes new routes possible, so it can miss the answer.
    """
    t0 = time.perf_counter()
    start, end = find(grid, "S"), find(grid, "E")
    init = (start[0], start[1], 0)
    parent = {init: None}
    dist = {init: 0}
    explored = []
    peak = 1
    goal = None

    if algo == "astar":
        er, ec = end
        h = lambda r, c: abs(r - er) + abs(c - ec)
        counter = 0
        heap = [(h(*start), 0, counter, init)]
        closed = set()
        while heap:
            _, neg_g, _, st = heapq.heappop(heap)
            if st in closed:
                continue
            closed.add(st)
            explored.append(st)
            if (st[0], st[1]) == end:
                goal = st
                break
            g = -neg_g + 1
            for nxt in neighbors(grid, st):
                if nxt not in closed and g < dist.get(nxt, 1 << 30):
                    dist[nxt] = g
                    parent[nxt] = st
                    counter += 1
                    heapq.heappush(heap, (g + h(nxt[0], nxt[1]), -g, counter, nxt))
            peak = max(peak, len(heap))
    elif algo == "naive":
        seen = {(start[0], start[1])}
        queue = deque([init])
        while queue:
            st = queue.popleft()
            explored.append(st)
            if (st[0], st[1]) == end:
                goal = st
                break
            for nxt in neighbors(grid, st):
                if (nxt[0], nxt[1]) in seen:
                    continue
                seen.add((nxt[0], nxt[1]))
                dist[nxt] = dist[st] + 1
                parent[nxt] = st
                queue.append(nxt)
            peak = max(peak, len(queue))
    else:
        algo = "bfs"
        queue = deque([init])
        while queue:
            st = queue.popleft()
            explored.append(st)
            if (st[0], st[1]) == end:
                goal = st
                break
            for nxt in neighbors(grid, st):
                if nxt in dist:
                    continue
                dist[nxt] = dist[st] + 1
                parent[nxt] = st
                queue.append(nxt)
            peak = max(peak, len(queue))

    elapsed = (time.perf_counter() - t0) * 1000
    states = []
    cur = goal
    while cur is not None:
        states.append(cur)
        cur = parent[cur]
    states.reverse()

    n = len(grid)
    key_types = sorted({ch for row in grid for ch in row if ch in KEYS} |
                       {ch.lower() for row in grid for ch in row if ch in DOORS})
    return {
        "algo": algo,
        "distance": dist[goal] if goal else -1,
        "states": [list(s) for s in states],
        "explored": [list(s) for s in explored[:EXPLORE_CAP]],
        "explored_total": len(explored),
        "peak_frontier": peak,
        "time_ms": round(elapsed, 3),
        "layers": len({s[2] for s in explored}),
        "key_types": "".join(key_types),
        "bound": n * n * (1 << len(key_types)),
    }


def path_grid(grid, path):
    """Copy of the grid with ordinary route cells marked x."""
    on_path = {tuple(p[:2]) for p in path}
    return [["x" if (r, c) in on_path and ch == "." else ch
             for c, ch in enumerate(row)] for r, row in enumerate(grid)]


# ============================================================== generator
def reachable_without(grid, door):
    """Can S reach E with `door` turned into a wall and every other door open?"""
    start, end = find(grid, "S"), find(grid, "E")
    n = len(grid)
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in seen \
                    and grid[nr][nc] not in ("#", door):
                seen.add((nr, nc))
                queue.append((nr, nc))
    return end in seen


def generate_maze(n, t, seed=None, loops=False):
    """Carve a real corridor maze and hide t key/door pairs in it.

    Rooms sit on odd rows/columns, walls on even ones, so the page can draw
    thin walls. A recursive backtracker carves a perfect maze (exactly one
    route S -> E). Doors go on that route, and each key is hidden deep in a
    side branch before its door, so reaching it means a detour and a walk
    back. With loops=True some walls are knocked out to create alternative
    routes, but only where every door still cannot be bypassed.
    """
    if not (1 <= t <= 5):
        raise ValueError("Keys must be between 1 and 5 (a door E would clash with the exit E).")
    if n % 2 == 0:
        n += 1
    if not (7 <= n <= 61):
        raise ValueError("Size must be between 7 and 61.")
    rng = random.Random(seed)
    grid = [["#"] * n for _ in range(n)]
    start, end = (1, 1), (n - 2, n - 2)

    parent = {start: None}
    stack = [start]
    grid[1][1] = "."
    while stack:
        r, c = stack[-1]
        options = [(r + dr, c + dc) for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
                   if 0 < r + dr < n - 1 and 0 < c + dc < n - 1 and (r + dr, c + dc) not in parent]
        if not options:
            stack.pop()
            continue
        nr, nc = rng.choice(options)
        grid[(r + nr) // 2][(c + nc) // 2] = "."
        grid[nr][nc] = "."
        parent[(nr, nc)] = (r, c)
        stack.append((nr, nc))

    route, cur = [], end
    while cur is not None:
        route.append(cur)
        cur = parent[cur]
    route.reverse()
    if len(route) < t + 2:
        raise ValueError("This maze is too small for that many keys. Increase the size.")
    on_route = {room: i for i, room in enumerate(route)}

    children = {}
    for room, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(room)
    attach, depth = {}, {}
    for i, room in enumerate(route):
        todo = [(ch, 1) for ch in children.get(room, []) if ch not in on_route]
        while todo:
            rm, d = todo.pop()
            attach[rm], depth[rm] = i, d
            todo.extend((ch, d + 1) for ch in children.get(rm, []))

    door_idx = sorted({min(len(route) - 2, max(1, round((i + 1) * (len(route) - 1) / (t + 1))))
                       for i in range(t)})
    while len(door_idx) < t:
        door_idx = sorted(set(door_idx) | {next(j for j in range(1, len(route) - 1) if j not in door_idx)})

    used = {start, end}
    prev = 0
    for key, d in zip(GEN_KEYS[:t], door_idx):
        cands = [rm for rm, a in attach.items() if prev <= a < d and rm not in used]
        if not cands:
            cands = [rm for rm, a in attach.items() if a < d and rm not in used]
        if cands:
            deepest = max(depth[rm] for rm in cands)
            key_room = rng.choice([rm for rm in cands if depth[rm] >= deepest - 2])
        else:
            spare = [route[j] for j in range(d - 1, 0, -1) if route[j] not in used]
            if not spare:
                raise ValueError("This maze is too small for that many keys. Increase the size.")
            key_room = spare[0]
        door_room = route[d]
        used |= {key_room, door_room}
        grid[key_room[0]][key_room[1]] = key
        grid[door_room[0]][door_room[1]] = key.upper()
        prev = d

    grid[start[0]][start[1]] = "S"
    grid[end[0]][end[1]] = "E"

    if loops:
        doors = [ch for row in grid for ch in row if ch in DOORS]
        walls = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                 if grid[r][c] == "#" and (r % 2) != (c % 2)]
        rng.shuffle(walls)
        target = max(2, len(walls) // 9)
        added = 0
        for r, c in walls:
            if added >= target:
                break
            grid[r][c] = "."
            if any(reachable_without(grid, d) for d in doors):
                grid[r][c] = "#"            # this loop would sneak around a door
            else:
                added += 1
    return grid


# ============================================================== web API
app = Flask(__name__)


def bad(msg):
    return jsonify(error=msg), 400


def grid_from_request():
    grid = (request.get_json(force=True, silent=True) or {}).get("grid")
    validate_grid(grid)
    return grid


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.post("/api/generate")
def api_generate():
    d = request.get_json(force=True, silent=True) or {}
    try:
        seed = d.get("seed")
        grid = generate_maze(int(d.get("n", 21)), int(d.get("t", 3)),
                             int(seed) if seed not in (None, "") else None,
                             bool(d.get("loops")))
    except (ValueError, TypeError) as e:
        return bad(str(e))
    return jsonify(grid=grid)


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if f is None:
        return bad("Choose a CSV file to upload.")
    try:
        text = f.read().decode("utf-8-sig")
        grid = [[c.strip() for c in row] for row in csv.reader(io.StringIO(text)) if any(row)]
        validate_grid(grid)
    except (ValueError, UnicodeDecodeError) as e:
        return bad(str(e))
    return jsonify(grid=grid)


@app.post("/api/solve")
def api_solve():
    try:
        grid = grid_from_request()
    except ValueError as e:
        return bad(str(e))
    algo = (request.get_json(force=True, silent=True) or {}).get("algo", "bfs")
    return jsonify(solve(grid, algo))


@app.post("/api/compare")
def api_compare():
    try:
        grid = grid_from_request()
    except ValueError as e:
        return bad(str(e))
    rows = []
    for algo in ("bfs", "astar", "naive"):
        res = solve(grid, algo)
        for k in ("states", "explored"):
            res.pop(k)
        rows.append(res)
    return jsonify(rows=rows)


@app.post("/api/export")
def api_export():
    try:
        grid = grid_from_request()
    except ValueError as e:
        return bad(str(e))
    res = solve(grid, "bfs")
    out = io.StringIO()
    csv.writer(out).writerows(path_grid(grid, res["states"]) if res["states"] else grid)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=shortest_path.csv"})


# ============================================================== frontend
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lockstep</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,500;12..96,600;12..96,800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --night: #141936;
    --deep: #0f1330;
    --panel: #1c2247;
    --line: #2f3769;
    --chalk: #eceaf8;
    --muted: #a3a8cc;
    --floor: #e6ebf6;
    --wall: #1b2044;
    --scan: #ffb547;
    --route: #6e8bff;
    --good: #3ddc97;
    --warn: #ff6b6b;
    --ka: #f4c430; --kb: #3ec7e0; --kc: #e2559b; --kd: #7bd34e; --ke: #a78bfa; --kf: #ff7f50;
    --sans: "Bricolage Grotesque", ui-sans-serif, system-ui, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html { background: var(--night); }
  body { margin: 0; color: var(--chalk); font-family: var(--sans); font-size: 16px; line-height: 1.5;
         background:
           radial-gradient(1200px 600px at 15% -10%, #24306b55, transparent 70%),
           var(--night); min-height: 100vh; }
  button, input, select { font: inherit; color: inherit; }
  :focus-visible { outline: 2px solid var(--scan); outline-offset: 2px; }

  /* ---------- header ---------- */
  .top { display: flex; align-items: center; gap: 18px; padding: 22px clamp(16px, 4vw, 40px) 8px; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand h1 { margin: 0; font-size: clamp(28px, 4vw, 40px); font-weight: 800; letter-spacing: -0.03em; line-height: 1; }
  .brand svg { width: 40px; height: 40px; flex: none; }
  .tag { margin: 0; color: var(--muted); max-width: 52ch; font-size: 15px; }

  /* ---------- main stage ---------- */
  .stage { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: clamp(16px, 2.5vw, 32px);
           padding: 14px clamp(16px, 4vw, 40px) 8px; align-items: start; }
  .board-col { min-width: 0; }
  #boardWrap { position: relative; width: 100%; max-width: 760px; }
  #board { display: block; border-radius: 6px; box-shadow: 0 0 0 1px #ffffff14, 0 30px 80px -30px #000c; touch-action: none; }
  #board.editing { cursor: crosshair; }
  #toast { position: absolute; left: 50%; top: 14px; transform: translateX(-50%); background: var(--deep);
           border: 1px solid var(--line); padding: 8px 14px; border-radius: 999px; font-size: 14px;
           opacity: 0; pointer-events: none; transition: opacity .2s; white-space: nowrap; max-width: 92%;
           overflow: hidden; text-overflow: ellipsis; }
  #toast.show { opacity: 1; }
  #banner { margin: 14px 0 0; font-size: 18px; font-weight: 600; min-height: 1.5em; max-width: 760px; }
  #banner.fail { color: var(--warn); }
  #banner.win { color: var(--good); }

  .readout { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); max-width: 760px; margin-top: 12px;
             border-top: 1px solid var(--line); }
  .readout div { padding: 10px 12px 4px 0; }
  .readout b { display: block; font-size: clamp(20px, 2.6vw, 28px); font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
  .readout span { color: var(--muted); font-size: 13px; }

  /* ---------- side rail ---------- */
  .rail { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
  .tabs { display: grid; grid-template-columns: repeat(3, 1fr); border-bottom: 1px solid var(--line); }
  .tabs button { background: none; border: 0; padding: 13px 0; cursor: pointer; color: var(--muted); font-weight: 600; font-size: 15px;
                 border-bottom: 2px solid transparent; }
  .tabs button[aria-selected="true"] { color: var(--chalk); border-bottom-color: var(--scan); }
  .pane { padding: 16px 18px 18px; display: none; }
  .pane.on { display: block; }
  .pane h3 { margin: 18px 0 8px; font-size: 14px; font-weight: 600; color: var(--muted); }
  .pane h3:first-child { margin-top: 0; }
  .row { display: flex; gap: 10px; align-items: end; flex-wrap: wrap; }
  .field { display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--muted); }
  .field input[type=number] { width: 76px; background: var(--deep); border: 1px solid var(--line); border-radius: 8px; padding: 7px 8px; }
  .seg { display: inline-flex; background: var(--deep); border: 1px solid var(--line); border-radius: 9px; padding: 3px; }
  .seg button { background: none; border: 0; padding: 6px 11px; border-radius: 7px; cursor: pointer; color: var(--muted); font-size: 14px; }
  .seg button[aria-pressed="true"] { background: var(--line); color: var(--chalk); }
  .btn { background: var(--deep); border: 1px solid var(--line); border-radius: 9px; padding: 8px 14px; cursor: pointer; font-weight: 500; }
  .btn:hover { border-color: #4a5394; }
  .btn.go { background: var(--scan); color: #1a1405; border-color: var(--scan); font-weight: 600; }
  .btn.go:hover { filter: brightness(1.06); }
  .btn:disabled { opacity: .45; cursor: default; }
  .btn.icon { padding: 8px 10px; }
  .full { width: 100%; }
  .ctrl { display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr; gap: 6px; margin-top: 14px; }
  .ctrl .btn { padding: 8px 6px; }
  .stack { display: flex; flex-direction: column; gap: 10px; }
  .hint { color: var(--muted); font-size: 13px; margin: 6px 0 0; }
  input[type=file] { display: none; }

  .algos { display: flex; flex-direction: column; gap: 8px; }
  .algo { display: grid; grid-template-columns: 18px 1fr; gap: 10px; align-items: start; padding: 10px 12px; border: 1px solid var(--line);
          border-radius: 10px; cursor: pointer; background: var(--deep); }
  .algo:has(input:checked) { border-color: var(--scan); }
  .algo input { margin: 4px 0 0; accent-color: var(--scan); }
  .algo strong { font-weight: 600; display: block; }
  .algo small { color: var(--muted); font-size: 13px; line-height: 1.4; display: block; }
  .speed { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); }
  .speed input { flex: 1; accent-color: var(--scan); }
  .check { display: flex; gap: 8px; align-items: center; font-size: 14px; color: var(--muted); cursor: pointer; }
  .check input { accent-color: var(--scan); }

  .tools { display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; }
  .tool { aspect-ratio: 1; border-radius: 8px; border: 1px solid var(--line); background: var(--deep); cursor: pointer;
          display: grid; place-items: center; font-weight: 700; font-size: 15px; padding: 0; }
  .tool[aria-pressed="true"] { outline: 2px solid var(--scan); outline-offset: 1px; }

  .pad { display: grid; grid-template-columns: repeat(3, 52px); grid-template-rows: repeat(2, 52px); gap: 6px; justify-content: center; margin-top: 12px; }
  .pad button { border-radius: 10px; border: 1px solid var(--line); background: var(--deep); cursor: pointer; font-size: 20px; }
  .pad .up { grid-column: 2; }
  .pad .left { grid-column: 1; grid-row: 2; } .pad .down { grid-column: 2; grid-row: 2; } .pad .right { grid-column: 3; grid-row: 2; }
  .inv { display: flex; gap: 6px; flex-wrap: wrap; min-height: 30px; align-items: center; }
  .chip { display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px 2px 6px; border-radius: 999px; background: var(--deep);
          border: 1px solid var(--line); font-size: 13px; font-weight: 600; }
  .chip i { width: 11px; height: 11px; transform: rotate(45deg); border-radius: 2px; display: inline-block; }

  .logbox { border-top: 1px solid var(--line); padding: 14px 18px 16px; }
  .logbox h3 { margin: 0 0 8px; font-size: 14px; font-weight: 600; color: var(--muted); }
  #log { list-style: none; margin: 0; padding: 0; max-height: 210px; overflow: auto; font-size: 14px; display: flex; flex-direction: column; gap: 6px; }
  #log li { display: grid; grid-template-columns: 52px 1fr; gap: 8px; }
  #log li em { font-style: normal; color: var(--muted); font-variant-numeric: tabular-nums; }
  #log li.empty { display: block; color: var(--muted); }

  /* ---------- key layers ---------- */
  .sec { padding: 26px clamp(16px, 4vw, 40px) 8px; }
  .sec h2 { margin: 0; font-size: clamp(22px, 2.6vw, 30px); font-weight: 800; letter-spacing: -0.025em; }
  .sec > p { margin: 6px 0 16px; color: var(--muted); max-width: 70ch; }
  #layers { display: flex; gap: 14px; overflow-x: auto; padding: 4px 2px 14px; scroll-snap-type: x proximity; }
  #layers .empty { color: var(--muted); padding: 22px 0; }
  .layer { flex: none; width: 150px; scroll-snap-align: start; transition: transform .25s; }
  .layer canvas { width: 150px; height: 150px; image-rendering: pixelated; border-radius: 6px; display: block;
                  box-shadow: 0 0 0 1px var(--line); transition: box-shadow .2s; }
  .layer.active canvas { box-shadow: 0 0 0 3px var(--route), 0 12px 30px -10px #6e8bffaa; }
  .layer.active { transform: translateY(-4px); }
  .layer .cap { display: flex; justify-content: space-between; align-items: baseline; margin-top: 8px; gap: 6px; }
  .layer code { font-family: var(--mono); font-size: 14px; font-weight: 700; letter-spacing: .04em; }
  .layer code b { font-weight: 700; }
  .layer small { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
  .layer .keys { display: flex; gap: 4px; margin-top: 4px; min-height: 18px; color: var(--muted); font-size: 12px; }

  /* ---------- compare ---------- */
  .cmp-wrap { overflow-x: auto; max-width: 900px; }
  table { border-collapse: collapse; width: 100%; min-width: 620px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 10px 14px 10px 0; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; font-size: 13px; }
  td.num, th.num { text-align: right; }
  td .bar { height: 6px; border-radius: 3px; background: var(--scan); margin-top: 6px; min-width: 2px; }
  tr.wrong td { color: var(--warn); }
  tr.wrong td .bar { background: var(--warn); }
  #cmpNote { color: var(--muted); max-width: 75ch; }
  .complexity { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 18px 28px; max-width: 1100px; margin-top: 8px; }
  .complexity div { border-left: 2px solid var(--line); padding-left: 14px; }
  .complexity b { display: block; font-weight: 600; }
  .complexity code { font-family: var(--mono); font-size: 14px; color: var(--scan); }
  .complexity p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
  footer { padding: 28px clamp(16px, 4vw, 40px) 40px; color: var(--muted); font-size: 13px; }

  @media (max-width: 960px) {
    .stage { grid-template-columns: 1fr; }
    .readout { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  }
  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }
</style>
</head>
<body>
<header class="top">
  <div class="brand">
    <svg viewBox="0 0 40 40" aria-hidden="true">
      <rect x="2" y="2" width="36" height="36" rx="7" fill="#1c2247" stroke="#2f3769"/>
      <path d="M9 9h22v8H17v6h14v8H9" fill="none" stroke="#e6ebf6" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
      <rect x="25.5" y="10.5" width="7" height="7" rx="1.5" transform="rotate(45 29 14)" fill="#f4c430"/>
    </svg>
    <h1>Lockstep</h1>
  </div>
  <p class="tag">Find the shortest way out of a maze where doors only open for the right key. The solver searches every cell once for each set of keys you could be holding.</p>
</header>

<main class="stage">
  <section class="board-col">
    <div id="boardWrap">
      <canvas id="board" tabindex="0" aria-label="Maze board"></canvas>
      <div id="toast" role="status"></div>
    </div>
    <p id="banner" aria-live="polite"></p>
    <div class="readout" aria-label="Search statistics">
      <div><b id="stPath">–</b><span>route length</span></div>
      <div><b id="stStates">–</b><span>states explored</span></div>
      <div><b id="stLayers">–</b><span>key layers used</span></div>
      <div><b id="stFrontier">–</b><span>peak queue size</span></div>
      <div><b id="stTime">–</b><span>solver time</span></div>
    </div>
  </section>

  <aside class="rail">
    <div class="tabs" role="tablist">
      <button role="tab" data-tab="build" aria-selected="false">Build</button>
      <button role="tab" data-tab="solve" aria-selected="true">Solve</button>
      <button role="tab" data-tab="play" aria-selected="false">Play</button>
    </div>

    <div class="pane" id="pane-build" role="tabpanel">
      <h3>New maze</h3>
      <div class="row">
        <label class="field">Size <input id="inN" type="number" value="21" min="7" max="61" step="2"></label>
        <label class="field">Keys <input id="inT" type="number" value="3" min="1" max="5"></label>
        <label class="field">Seed <input id="inSeed" type="number" value="2026"></label>
        <button class="btn icon" id="dice" title="Random seed" aria-label="Random seed">⚄</button>
      </div>
      <div class="row" style="margin-top:12px">
        <div class="seg" role="group" aria-label="Maze layout">
          <button aria-pressed="true" data-loops="0">One route</button>
          <button aria-pressed="false" data-loops="1">With loops</button>
        </div>
      </div>
      <p class="hint" id="loopHint">Every door sits on the only route out, so each key is required.</p>
      <div class="stack" style="margin-top:12px">
        <button class="btn go full" id="genBtn">Generate maze</button>
      </div>

      <h3>Your own maze</h3>
      <div class="row">
        <label class="btn" for="file">Upload CSV</label>
        <input type="file" id="file" accept=".csv,text/csv">
        <button class="btn" id="dlMaze">Download maze CSV</button>
      </div>
      <p class="hint">N × N grid using . # S E, keys a–f and doors A–F.</p>

      <h3>Edit by hand</h3>
      <label class="check"><input type="checkbox" id="editOn"> Paint on the board</label>
      <div class="tools" id="tools" style="margin-top:10px"></div>
      <p class="hint">Click or drag on the maze. Start and exit move instead of copying.</p>
    </div>

    <div class="pane on" id="pane-solve" role="tabpanel">
      <h3>Algorithm</h3>
      <div class="algos">
        <label class="algo"><input type="radio" name="algo" value="bfs" checked>
          <span><strong>BFS with key bitmask</strong><small>Explores every (cell, keys held) state in order of distance. Always finds the shortest route.</small></span></label>
        <label class="algo"><input type="radio" name="algo" value="astar">
          <span><strong>A* with key bitmask</strong><small>Same states, but tries cells closer to the exit first. Same answer, usually fewer states.</small></span></label>
        <label class="algo"><input type="radio" name="algo" value="naive">
          <span><strong>BFS that forgets keys</strong><small>Remembers only cells. Breaks when a key forces you back the way you came. Here to show why the bitmask matters.</small></span></label>
      </div>
      <h3>Playback</h3>
      <div class="speed">Slow <input type="range" id="speed" min="1" max="10" value="6" aria-label="Speed"> Fast</div>
      <label class="check" style="margin-top:10px"><input type="checkbox" id="showSearch" checked> Show the search, not just the route</label>
      <div class="ctrl">
        <button class="btn go" id="runBtn">Solve</button>
        <button class="btn" id="pauseBtn" disabled>Pause</button>
        <button class="btn" id="stepBtn" disabled>Step</button>
        <button class="btn" id="endBtn" disabled>Finish</button>
      </div>
      <div class="row" style="margin-top:10px">
        <button class="btn" id="dlPath">Download route CSV</button>
      </div>
    </div>

    <div class="pane" id="pane-play" role="tabpanel">
      <h3>Beat the algorithm</h3>
      <p class="hint" style="margin-top:0">Walk to the exit with the arrow keys or WASD. Collect keys, open doors, and try to match the shortest route.</p>
      <div class="row" style="margin-top:12px">
        <button class="btn go" id="playBtn">Start walking</button>
        <button class="btn" id="hintBtn" disabled>Show next move</button>
      </div>
      <h3>Keys held</h3>
      <div class="inv" id="inv"><span class="hint" style="margin:0">None yet</span></div>
      <h3>Steps</h3>
      <div style="font-size:28px;font-weight:600;font-variant-numeric:tabular-nums" id="playSteps">0</div>
      <div class="pad" aria-label="Move">
        <button class="up" data-move="-1,0" aria-label="Up">↑</button>
        <button class="left" data-move="0,-1" aria-label="Left">←</button>
        <button class="down" data-move="1,0" aria-label="Down">↓</button>
        <button class="right" data-move="0,1" aria-label="Right">→</button>
      </div>
    </div>

    <div class="logbox">
      <h3>Route log</h3>
      <ol id="log"><li class="empty">Solve or walk the maze to see each key and door along the way.</li></ol>
    </div>
  </aside>
</main>

<section class="sec">
  <h2>Key layers</h2>
  <p>Holding different keys changes where you can go, so the solver treats each set of keys as its own copy of the maze. Each copy below shows where the search reached while holding exactly those keys. Picking up a key lifts the route to another copy.</p>
  <div id="layers"><div class="empty">Run a search to see its layers appear.</div></div>
</section>

<section class="sec">
  <h2>Compare algorithms</h2>
  <p>Run all three solvers on the current maze and see how much work each one does.</p>
  <button class="btn go" id="cmpBtn">Run all three</button>
  <div class="cmp-wrap" style="margin-top:14px"><table id="cmp" hidden></table></div>
  <p id="cmpNote"></p>

  <div class="complexity">
    <div><b>State</b><code>(row, col, mask)</code><p>mask is a T-bit number. Bit i is 1 when key i is in hand.</p></div>
    <div><b>Time and memory</b><code>O(N² · 2^T)</code><p>Each cell can be visited once per key set, and each visit checks 4 neighbours.</p></div>
    <div><b>Why BFS is optimal</b><code>every move costs 1</code><p>States leave the queue in order of distance, so the first time the exit appears is the shortest.</p></div>
    <div><b>This maze</b><code id="boundCode">–</code><p id="boundText">The most states any solver could need on the current maze.</p></div>
  </div>
</section>

<footer>Lockstep. Built for the DSA hackathon problem "Maze with Keys and Doors". Python + Flask backend, canvas frontend, no other dependencies.</footer>

<script>
"use strict";
const KEYS = "abcdef";
const KCOL = { a: "#f4c430", b: "#3ec7e0", c: "#e2559b", d: "#7bd34e", e: "#a78bfa", f: "#ff7f50" };
const C = { wall: "#1b2044", floor: "#e6ebf6", scan: "255,181,71", route: "#6e8bff", ink: "#1b2044",
            start: "#3ddc97", exit: "#ff5d6c" };
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const S = {
  grid: [], n: 0, lattice: false,
  mode: "solve", tool: "#", editing: false, loops: false,
  run: null,      // solver playback
  play: null,     // manual walking
  hint: null,
};

/* ---------------- helpers ---------------- */
const bitOf = ch => 1 << (ch.toLowerCase().charCodeAt(0) - 97);
const keysOf = m => [...KEYS].filter((k, i) => (m >> i) & 1);
function keyTypes() {
  const s = new Set();
  for (const row of S.grid) for (const ch of row) if (/[a-fA-DF]/.test(ch)) s.add(ch.toLowerCase());
  return [...s].sort();
}
function maskBits(m) {
  const kt = keyTypes();
  const width = kt.length ? KEYS.indexOf(kt[kt.length - 1]) + 1 : 1;
  return m.toString(2).padStart(width, "0");
}
function findCell(ch) {
  for (let r = 0; r < S.n; r++) for (let c = 0; c < S.n; c++) if (S.grid[r][c] === ch) return [r, c];
  return null;
}
function isLattice(g) {
  const n = g.length;
  if (n < 5 || n % 2 === 0) return false;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const ch = g[r][c], edge = r === 0 || c === 0 || r === n - 1 || c === n - 1;
    if ((edge || (r % 2 === 0 && c % 2 === 0)) && ch !== "#") return false;
    if (r % 2 === 1 && c % 2 === 1 && ch === "#") return false;
  }
  return true;
}
const isRoom = (r, c) => !S.lattice || (r % 2 === 1 && c % 2 === 1);
const fmt = x => x.toLocaleString("en-IN");

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
}
function banner(msg, kind = "") { const b = $("#banner"); b.textContent = msg; b.className = kind; }
function log(step, html) {
  const ol = $("#log");
  if (ol.querySelector(".empty")) ol.innerHTML = "";
  const li = document.createElement("li");
  li.innerHTML = `<em>${step === null ? "" : "Step " + step}</em><span>${html}</span>`;
  ol.appendChild(li); ol.scrollTop = ol.scrollHeight;
}
function clearLog(msg) { $("#log").innerHTML = `<li class="empty">${msg}</li>`; }
const keyChip = k => `<span class="chip"><i style="background:${KCOL[k]}"></i>${k}</span>`;

async function api(url, body, isForm = false) {
  const res = await fetch(url, isForm ? { method: "POST", body } :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("json") ? await res.json() : await res.blob();
  if (!res.ok) { const m = data.error || "Something went wrong on the server."; toast(m); banner(m, "fail"); throw new Error(m); }
  return data;
}

/* ---------------- board geometry ---------------- */
const cv = $("#board"), ctx = cv.getContext("2d");
let G = null;
function layout() {
  if (!S.n) return;
  const W = Math.floor(Math.min($("#boardWrap").clientWidth, 760));
  const w = i => (S.lattice && i % 2 === 0 ? 0.3 : 1);
  let total = 0;
  for (let i = 0; i < S.n; i++) total += w(i);
  const u = W / total, edges = [0];
  let acc = 0;
  for (let i = 0; i < S.n; i++) { acc += w(i); edges.push(Math.round(acc * u)); }
  G = { W, u, x: edges.slice(0, S.n), s: Array.from({ length: S.n }, (_, i) => edges[i + 1] - edges[i]) };
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = cv.style.height = W + "px";
  cv.width = cv.height = Math.round(W * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
const cx = c => G.x[c] + G.s[c] / 2;
const cy = r => G.x[r] + G.s[r] / 2;
function cellAt(px, py) {
  const pick = v => { for (let i = S.n - 1; i >= 0; i--) if (v >= G.x[i]) return i; return 0; };
  return [pick(py), pick(px)];
}

/* ---------------- drawing ---------------- */
function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h); }

function drawKey(x, y, s, k) {
  const d = s * 0.30;
  ctx.save(); ctx.translate(x, y); ctx.rotate(Math.PI / 4);
  ctx.fillStyle = KCOL[k]; ctx.strokeStyle = C.ink; ctx.lineWidth = Math.max(1.2, s * 0.05);
  roundRect(-d, -d, d * 2, d * 2, d * 0.25); ctx.fill(); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = C.ink; ctx.font = `700 ${Math.max(8, s * 0.36)}px ${getComputedStyle(document.body).fontFamily}`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(k, x, y + s * 0.02);
}
function drawDoor(r, c, ch, open) {
  const x = G.x[c], y = G.x[r], w = G.s[c], h = G.s[r], col = KCOL[ch.toLowerCase()];
  if (open) {
    ctx.strokeStyle = col; ctx.lineWidth = Math.max(1.5, G.u * 0.07); ctx.setLineDash([G.u * 0.12, G.u * 0.1]);
    ctx.strokeRect(x + w * 0.12, y + h * 0.12, w * 0.76, h * 0.76); ctx.setLineDash([]);
    return;
  }
  ctx.fillStyle = col; ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#1b2044cc";
  const bars = 4;
  for (let i = 0; i < bars; i++) ctx.fillRect(x + w * (0.1 + i * 0.25), y, Math.max(1, w * 0.07), h);
  const s = Math.min(w, h);
  ctx.fillStyle = C.floor; roundRect(x + w / 2 - s * 0.2, y + h / 2 - s * 0.2, s * 0.4, s * 0.4, s * 0.08); ctx.fill();
  ctx.fillStyle = C.ink; ctx.font = `700 ${Math.max(8, s * 0.3)}px ${getComputedStyle(document.body).fontFamily}`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(ch, x + w / 2, y + h / 2 + s * 0.02);
}
function drawEnd(r, c, kind) {
  const x = cx(c), y = cy(r), s = Math.min(G.s[c], G.s[r]);
  if (kind === "S") {
    ctx.fillStyle = C.start; ctx.beginPath(); ctx.arc(x, y, s * 0.38, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = C.ink; ctx.font = `800 ${Math.max(8, s * 0.36)}px ${getComputedStyle(document.body).fontFamily}`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("S", x, y + s * 0.02);
  } else {
    for (const [rad, col] of [[0.42, C.exit], [0.29, C.floor], [0.16, C.exit]]) {
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, s * rad, 0, Math.PI * 2); ctx.fill();
    }
  }
}

function currentTrail() {
  if (S.play) return S.play.trail;
  if (S.run && S.run.ti >= 0) return S.run.trail.slice(0, S.run.ti + 1);
  return null;
}

function draw() {
  if (!G) return;
  const { grid, n } = S, run = S.run, play = S.play;
  ctx.fillStyle = C.wall; ctx.fillRect(0, 0, G.W, G.W);

  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    if (grid[r][c] === "#") continue;
    ctx.fillStyle = C.floor; ctx.fillRect(G.x[c], G.x[r], G.s[c], G.s[r]);
  }
  // search heat: deeper amber = cell explored in more key layers
  if (run && run.showSearch && run.ei > 0) {
    const maxL = Math.max(1, run.layerOrder.length);
    for (let i = 0; i < n * n; i++) {
      const h = run.heat[i];
      if (!h) continue;
      const r = (i / n) | 0, c = i % n;
      ctx.fillStyle = `rgba(${C.scan},${0.16 + 0.5 * Math.min(1, h / maxL)})`;
      ctx.fillRect(G.x[c], G.x[r], G.s[c], G.s[r]);
    }
    if (run.phase === "explore") {
      for (const [r, c] of run.recent) {
        ctx.fillStyle = `rgba(${C.scan},0.95)`;
        ctx.fillRect(G.x[c], G.x[r], G.s[c], G.s[r]);
      }
    }
  }
  // route, coloured by the last key picked up
  const trail = currentTrail();
  if (trail && trail.length > 1) {
    ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.lineWidth = Math.max(2, G.u * 0.3);
    for (let i = 1; i < trail.length; i++) {
      const a = trail[i - 1], b = trail[i];
      ctx.strokeStyle = b.k ? KCOL[b.k] : C.route;
      ctx.beginPath(); ctx.moveTo(cx(a.c), cy(a.r)); ctx.lineTo(cx(b.c), cy(b.r)); ctx.stroke();
    }
  }
  // items
  const taken = run ? run.taken : play ? play.taken : null;
  const opened = run ? run.opened : play ? play.opened : null;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const ch = grid[r][c], idx = r * n + c;
    if (ch === "S" || ch === "E") drawEnd(r, c, ch);
    else if (/[a-f]/.test(ch) && !(taken && taken.has(idx))) drawKey(cx(c), cy(r), Math.min(G.s[c], G.s[r]), ch);
    else if (/[A-DF]/.test(ch)) drawDoor(r, c, ch, opened && opened.has(idx));
  }
  // next-move hint
  if (S.hint) {
    const [r, c] = S.hint;
    ctx.strokeStyle = C.route; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cx(c), cy(r), Math.min(G.s[c], G.s[r]) * 0.45, 0, Math.PI * 2); ctx.stroke();
  }
  // walker
  const at = play ? play : (run && run.ti >= 0 ? run.trail[run.ti] : null);
  if (at) {
    const s = Math.min(G.s[at.c], G.s[at.r], G.u);
    ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(cx(at.c), cy(at.r), s * 0.34, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = (at.k ? KCOL[at.k] : C.route); ctx.beginPath(); ctx.arc(cx(at.c), cy(at.r), s * 0.25, 0, Math.PI * 2); ctx.fill();
  }
}

/* ---------------- key layers panel ---------------- */
function resetLayers(msg) {
  $("#layers").innerHTML = `<div class="empty">${msg || "Run a search to see its layers appear."}</div>`;
}
function addLayerCard(m) {
  const box = $("#layers");
  if (box.querySelector(".empty")) box.innerHTML = "";
  const el = document.createElement("div");
  el.className = "layer"; el.dataset.mask = m;
  const ks = keysOf(m);
  el.innerHTML = `<canvas width="${S.n}" height="${S.n}"></canvas>
    <div class="cap"><code>${maskBits(m)}</code><small data-count>0 states</small></div>
    <div class="keys">${ks.length ? ks.map(keyChip).join("") : "no keys"}</div>`;
  el.title = `Mask ${m} = binary ${maskBits(m)}`;
  box.appendChild(el);
  return el;
}
function paintLayer(m) {
  const run = S.run, L = run.layers.get(m), n = S.n;
  const canvas = L.el.querySelector("canvas"), c2 = canvas.getContext("2d");
  const img = c2.createImageData(n, n), d = img.data;
  const tint = keysOf(m).length ? hexRgb(KCOL[keysOf(m).slice(-1)[0]]) : [110, 139, 255];
  for (let i = 0; i < n * n; i++) {
    const ch = S.grid[(i / n) | 0][i % n];
    let rgb;
    if (ch === "#") rgb = [27, 32, 68];
    else if (L.route[i]) rgb = [255, 255, 255];
    else if (L.cells[i]) rgb = tint;
    else rgb = [70, 78, 128];
    d[i * 4] = rgb[0]; d[i * 4 + 1] = rgb[1]; d[i * 4 + 2] = rgb[2]; d[i * 4 + 3] = 255;
  }
  c2.putImageData(img, 0, 0);
  L.el.querySelector("[data-count]").textContent = `${fmt(L.count)} states`;
}
function hexRgb(h) { const v = parseInt(h.slice(1), 16); return [v >> 16, (v >> 8) & 255, v & 255]; }
function setActiveLayer(m) {
  $$(".layer").forEach(el => el.classList.toggle("active", +el.dataset.mask === m));
  const el = $(`.layer[data-mask="${m}"]`), box = $("#layers");
  if (el) {
    const left = el.offsetLeft - box.offsetLeft, right = left + el.offsetWidth;
    if (left < box.scrollLeft || right > box.scrollLeft + box.clientWidth)
      box.scrollTo({ left: Math.max(0, left - 20), behavior: reduceMotion ? "auto" : "smooth" });
  }
}

/* ---------------- loading a maze ---------------- */
function loadGrid(g, note) {
  stopAll();
  S.grid = g.map(row => [...row]); S.n = g.length; S.lattice = isLattice(S.grid);
  layout(); draw();
  resetLayers(); clearLog("Solve or walk the maze to see each key and door along the way.");
  setStats(null);
  $("#cmp").hidden = true; $("#cmpNote").textContent = "";
  const kt = keyTypes();
  $("#boundCode").textContent = `${S.n}² × 2^${kt.length} = ${fmt(S.n * S.n * 2 ** kt.length)}`;
  banner(note || `${S.n} × ${S.n} maze with ${kt.length} key ${kt.length === 1 ? "type" : "types"}.`);
}
function setStats(res, shown) {
  if (!res) { ["#stPath", "#stStates", "#stLayers", "#stFrontier", "#stTime"].forEach(s => $(s).textContent = "–"); return; }
  $("#stPath").textContent = res.distance < 0 ? "none" : fmt(res.distance);
  $("#stStates").textContent = fmt(shown ?? res.explored_total);
  $("#stLayers").textContent = `${res.layers} / ${2 ** res.key_types.length}`;
  $("#stFrontier").textContent = fmt(res.peak_frontier);
  $("#stTime").textContent = res.time_ms < 1 ? `${res.time_ms.toFixed(2)} ms` : `${res.time_ms.toFixed(1)} ms`;
}

async function generate() {
  const n = +$("#inN").value, t = +$("#inT").value;
  const data = await api("/api/generate", { n, t, seed: $("#inSeed").value, loops: S.loops });
  loadGrid(data.grid);
  return data;
}

/* ---------------- solver playback ---------------- */
function stopAll() {
  if (S.run) cancelAnimationFrame(S.run.raf);
  S.run = null; S.play = null; S.hint = null;
  $("#pauseBtn").disabled = $("#stepBtn").disabled = $("#endBtn").disabled = true;
  $("#pauseBtn").textContent = "Pause";
  $("#hintBtn").disabled = true;
}

function buildTrail(states) {
  // annotate each step with the colour key (last key picked up)
  const out = []; let k = null, mask = 0;
  for (const [r, c, m] of states) {
    if (m !== mask) { const gained = keysOf(m & ~mask); if (gained.length) k = gained[gained.length - 1]; mask = m; }
    out.push({ r, c, m, k });
  }
  return out;
}

async function runSolve() {
  if (S.editing) toggleEdit(false);
  stopAll();
  const algo = $("input[name=algo]:checked").value;
  const res = await api("/api/solve", { grid: S.grid, algo });
  const n = S.n;
  const run = S.run = {
    res, algo, phase: "explore", ei: 0, ti: -1, last: 0, paused: false,
    explored: res.explored, trail: buildTrail(res.states),
    heat: new Uint16Array(n * n), layers: new Map(), layerOrder: [], dirty: new Set(),
    recent: [], taken: new Set(), opened: new Set(),
    showSearch: $("#showSearch").checked, raf: 0, curMask: -1,
  };
  resetLayers(); clearLog("Searching…");
  banner(algo === "astar" ? "A* is searching, heading for the exit first…" :
         algo === "naive" ? "Searching while forgetting which keys are held…" : "BFS is searching outward, one step at a time…");
  setStats(res, 0);
  $("#stPath").textContent = "…";
  $("#pauseBtn").disabled = $("#stepBtn").disabled = $("#endBtn").disabled = false;
  if (!run.showSearch || reduceMotion) { exploreAll(); }
  run.raf = requestAnimationFrame(tick);
}

function exploreOne() {
  const run = S.run, n = S.n;
  const [r, c, m] = run.explored[run.ei++];
  let L = run.layers.get(m);
  if (!L) {
    L = { cells: new Uint8Array(n * n), route: new Uint8Array(n * n), count: 0, el: addLayerCard(m) };
    run.layers.set(m, L); run.layerOrder.push(m);
  }
  const i = r * n + c;
  if (!L.cells[i]) { L.cells[i] = 1; run.heat[i]++; }
  L.count++;
  run.dirty.add(m);
  return [r, c];
}
function exploreAll() {
  const run = S.run;
  while (run.ei < run.explored.length) exploreOne();
  run.recent = [];
  startTrace();
}
function startTrace() {
  const run = S.run;
  run.phase = "trace";
  setStats(run.res);
  if (run.res.distance < 0) { finishRun(); return; }
  clearLog("Following the route…");
  run.ti = 0; markRoute(0);
}
function markRoute(i) {
  const run = S.run, st = run.trail[i], L = run.layers.get(st.m);
  if (L) { L.route[st.r * S.n + st.c] = 1; run.dirty.add(st.m); }
  if (st.m !== run.curMask) { run.curMask = st.m; setActiveLayer(st.m); }
  const ch = S.grid[st.r][st.c], idx = st.r * S.n + st.c;
  if (/[a-f]/.test(ch) && !run.taken.has(idx)) {
    run.taken.add(idx);
    log(i, `Picked up ${keyChip(ch)}. Now searching layer <code>${maskBits(st.m)}</code>.`);
  } else if (/[A-DF]/.test(ch) && !run.opened.has(idx)) {
    run.opened.add(idx);
    log(i, `Opened door <b>${ch}</b> with ${keyChip(ch.toLowerCase())}.`);
  }
}
function traceStep() {
  const run = S.run;
  if (run.ti >= run.trail.length - 1) return false;
  do { run.ti++; markRoute(run.ti); }
  while (run.ti < run.trail.length - 1 && !isRoom(run.trail[run.ti].r, run.trail[run.ti].c));
  return true;
}
const SPEED_EXPLORE = [1, 1, 2, 3, 5, 8, 13, 22, 40, 80];
const SPEED_TRACE = [520, 400, 300, 230, 170, 125, 90, 62, 40, 22];

function tick(ts) {
  const run = S.run;
  if (!run) return;
  const sp = +$("#speed").value - 1;
  if (!run.paused) {
    if (run.phase === "explore") {
      const scale = Math.max(1, run.explored.length / 900);
      const k = Math.ceil(SPEED_EXPLORE[sp] * scale);
      const recent = [];
      for (let j = 0; j < k && run.ei < run.explored.length; j++) recent.push(exploreOne());
      run.recent = recent;
      $("#stStates").textContent = fmt(run.ei);
      if (run.ei >= run.explored.length) { run.recent = []; startTrace(); }
    } else if (run.phase === "trace" && ts - run.last > SPEED_TRACE[sp]) {
      run.last = ts;
      if (!traceStep()) { finishRun(); }
    }
  }
  flushLayers(); draw();
  if (S.run === run && run.phase !== "done") run.raf = requestAnimationFrame(tick);
}
function flushLayers() {
  const run = S.run;
  if (!run) return;
  for (const m of run.dirty) paintLayer(m);
  run.dirty.clear();
}
function finishRun() {
  const run = S.run;
  run.phase = "done";
  flushLayers(); draw();
  $("#pauseBtn").disabled = $("#stepBtn").disabled = $("#endBtn").disabled = true;
  const res = run.res, used = keysOf(run.trail.length ? run.trail[run.trail.length - 1].m : 0);
  if (res.distance < 0) {
    if (run.algo === "naive") {
      banner("The key-forgetting BFS found no way out. It marked cells as visited before it had the key, so it never went back through them.", "fail");
      log(null, "No route found by this solver. Try BFS with key bitmask on the same maze.");
    } else {
      banner("There is no way out. Every route to the exit is blocked by a door whose key cannot be reached.", "fail");
      log(null, "No route exists.");
    }
    return;
  }
  const truncated = res.explored_total > res.explored.length ? ` (showing the first ${fmt(res.explored.length)})` : "";
  banner(`Reached the exit in ${fmt(res.distance)} steps after exploring ${fmt(res.explored_total)} states${truncated}.`, "win");
  log(res.distance, `Reached the exit holding ${used.length ? used.map(keyChip).join(" ") : "no keys"}.`);
  if (run.algo === "naive") checkNaive(res.distance);
}
async function checkNaive(d) {
  const best = jsBfs(...findCell("S"), 0);
  if (best && best.dist < d) banner(`The key-forgetting BFS took ${d} steps, but the true shortest route is ${best.dist}. It skipped cells it had already seen without the key.`, "fail");
}
function togglePause() {
  const run = S.run; if (!run) return;
  run.paused = !run.paused; $("#pauseBtn").textContent = run.paused ? "Resume" : "Pause";
}
function stepOnce() {
  const run = S.run; if (!run) return;
  run.paused = true; $("#pauseBtn").textContent = "Resume";
  if (run.phase === "explore") {
    run.recent = [exploreOne()];
    $("#stStates").textContent = fmt(run.ei);
    if (run.ei >= run.explored.length) { run.recent = []; startTrace(); }
  } else if (run.phase === "trace") { if (!traceStep()) finishRun(); }
  flushLayers(); draw();
}
function skipToEnd() {
  const run = S.run; if (!run) return;
  if (run.phase === "explore") exploreAll();
  if (run.phase === "trace") { while (traceStep()) {} }
  finishRun();
}

/* ---------------- browser-side BFS (hints, play scoring) ---------------- */
function jsBfs(sr, sc, sm) {
  const g = S.grid, n = S.n, NN = n * n;
  const code = (r, c, m) => m * NN + r * n + c;
  const prev = new Map([[code(sr, sc, sm), -1]]);
  const q = [[sr, sc, sm]];
  for (let h = 0; h < q.length; h++) {
    const [r, c, m] = q[h];
    if (g[r][c] === "E") {
      const path = []; let k = code(r, c, m);
      while (k !== -1) { const mm = Math.floor(k / NN), rest = k % NN; path.push([(rest / n) | 0, rest % n, mm]); k = prev.get(k); }
      path.reverse();
      return { dist: path.length - 1, path };
    }
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
      const ch = g[nr][nc]; let nm = m;
      if (ch === "#") continue;
      if (/[A-DF]/.test(ch) && !(m & bitOf(ch))) continue;
      if (/[a-f]/.test(ch)) nm |= bitOf(ch);
      const k = code(nr, nc, nm);
      if (!prev.has(k)) { prev.set(k, code(r, c, m)); q.push([nr, nc, nm]); }
    }
  }
  return null;
}

/* ---------------- play mode ---------------- */
function startPlay() {
  if (S.editing) toggleEdit(false);
  stopAll();
  const s = findCell("S");
  if (!s) { toast("Place a start (S) first."); return; }
  const best = jsBfs(s[0], s[1], 0);
  S.play = { r: s[0], c: s[1], m: 0, k: null, steps: 0, trail: [{ r: s[0], c: s[1], m: 0, k: null }],
             taken: new Set(), opened: new Set(), done: false, best: best ? best.dist : -1 };
  $("#hintBtn").disabled = false;
  resetLayers("Layers come from a solver run. Switch to Solve to see them.");
  clearLog("Your walk will be logged here.");
  updateInv();
  banner(best ? `Find the exit. The shortest route is ${best.dist} steps.` : "Heads up: this maze has no way out. Have a look around anyway.");
  draw(); cv.focus();
}
function updateInv() {
  const p = S.play;
  const ks = p ? keysOf(p.m) : [];
  $("#inv").innerHTML = ks.length ? ks.map(keyChip).join("") : `<span class="hint" style="margin:0">None yet</span>`;
  $("#playSteps").textContent = p ? p.steps : 0;
}
function stepTo(nr, nc) {
  const p = S.play, n = S.n;
  if (nr < 0 || nc < 0 || nr >= n || nc >= n) return false;
  const ch = S.grid[nr][nc];
  if (ch === "#") return false;
  if (/[A-DF]/.test(ch) && !(p.m & bitOf(ch))) {
    toast(`Door ${ch} is locked. Find key ${ch.toLowerCase()} first.`);
    return false;
  }
  const idx = nr * n + nc;
  p.r = nr; p.c = nc; p.steps++;
  if (/[a-f]/.test(ch) && !(p.m & bitOf(ch))) {
    p.m |= bitOf(ch); p.k = ch; p.taken.add(idx);
    log(p.steps, `Picked up ${keyChip(ch)}.`);
    toast(`Picked up key ${ch}.`);
  } else if (/[a-f]/.test(ch)) { p.taken.add(idx); }
  if (/[A-DF]/.test(ch) && !p.opened.has(idx)) { p.opened.add(idx); log(p.steps, `Opened door <b>${ch}</b>.`); }
  p.trail.push({ r: nr, c: nc, m: p.m, k: p.k });
  return true;
}
function move(dr, dc) {
  const p = S.play;
  if (!p || p.done) return;
  S.hint = null;
  const ok = stepTo(p.r + dr, p.c + dc);
  // in the thin-wall view, carry on through the narrow passage into the next room
  if (ok && S.lattice && !isRoom(p.r, p.c) && S.grid[p.r][p.c] === ".") stepTo(p.r + dr, p.c + dc);
  updateInv(); draw();
  if (S.grid[p.r][p.c] === "E") {
    p.done = true; $("#hintBtn").disabled = true;
    const diff = p.steps - p.best;
    log(p.steps, "Reached the exit.");
    banner(diff === 0 ? `You escaped in ${p.steps} steps. That is exactly the shortest route.`
                      : `You escaped in ${p.steps} steps. The shortest route is ${p.best}, so you took ${diff} extra.`, diff === 0 ? "win" : "");
  }
}
function showHint() {
  const p = S.play; if (!p || p.done) return;
  const res = jsBfs(p.r, p.c, p.m);
  if (!res || res.path.length < 2) { toast("No route from here."); return; }
  let i = 1;
  if (S.lattice && !isRoom(res.path[1][0], res.path[1][1]) && res.path.length > 2) i = 2;
  S.hint = [res.path[i][0], res.path[i][1]];
  toast(`${res.dist} steps to go from here.`);
  draw();
}

/* ---------------- editor ---------------- */
const TOOLS = [["#", "Wall"], [".", "Floor"], ["S", "Start"], ["E", "Exit"],
  ["a", "Key a"], ["b", "Key b"], ["c", "Key c"], ["d", "Key d"], ["f", "Key f"],
  ["A", "Door A"], ["B", "Door B"], ["C", "Door C"], ["D", "Door D"], ["F", "Door F"]];
function buildTools() {
  const box = $("#tools");
  for (const [ch, label] of TOOLS) {
    const b = document.createElement("button");
    b.className = "tool"; b.dataset.tool = ch; b.title = label; b.setAttribute("aria-label", label);
    b.setAttribute("aria-pressed", ch === S.tool);
    if (ch === "#") { b.style.background = C.wall; b.style.borderColor = "#4a5394"; }
    else if (ch === ".") { b.style.background = C.floor; }
    else if (ch === "S") { b.style.background = C.start; b.style.color = C.ink; b.textContent = "S"; }
    else if (ch === "E") { b.style.background = C.exit; b.style.color = "#fff"; b.textContent = "E"; }
    else if (/[a-f]/.test(ch)) { b.innerHTML = `<span style="width:16px;height:16px;transform:rotate(45deg);background:${KCOL[ch]};border-radius:3px;display:block"></span>`; b.title = label; }
    else { b.style.background = `repeating-linear-gradient(90deg, ${KCOL[ch.toLowerCase()]} 0 5px, #1b2044 5px 7px)`; b.style.color = C.ink; b.innerHTML = `<span style="background:${C.floor};border-radius:3px;padding:0 4px">${ch}</span>`; }
    b.onclick = () => { S.tool = ch; $$(".tool").forEach(t => t.setAttribute("aria-pressed", t === b)); if (!S.editing) toggleEdit(true); };
    box.appendChild(b);
  }
}
function toggleEdit(on) {
  S.editing = on; $("#editOn").checked = on;
  cv.classList.toggle("editing", on);
  if (on) { stopAll(); resetLayers(); setStats(null); banner("Painting. Click or drag on the maze."); draw(); }
}
let painting = false;
function paintAt(ev) {
  const rect = cv.getBoundingClientRect();
  const [r, c] = cellAt(ev.clientX - rect.left, ev.clientY - rect.top);
  const t = S.tool;
  if (S.grid[r][c] === t) return;
  if (t === "S" || t === "E") { const old = findCell(t); if (old) S.grid[old[0]][old[1]] = "."; }
  if ((S.grid[r][c] === "S" || S.grid[r][c] === "E") && t !== S.grid[r][c] && painting === "drag") return;
  S.grid[r][c] = t;
  draw();
  const kt = keyTypes();
  $("#boundCode").textContent = `${S.n}² × 2^${kt.length} = ${fmt(S.n * S.n * 2 ** kt.length)}`;
}
cv.addEventListener("pointerdown", e => { if (!S.editing) return; painting = "down"; cv.setPointerCapture(e.pointerId); paintAt(e); painting = "drag"; });
cv.addEventListener("pointermove", e => { if (S.editing && painting) paintAt(e); });
cv.addEventListener("pointerup", () => { painting = false; });

/* ---------------- compare ---------------- */
const NAMES = { bfs: "BFS with key bitmask", astar: "A* with key bitmask", naive: "BFS that forgets keys" };
async function compare() {
  const data = await api("/api/compare", { grid: S.grid });
  const rows = data.rows, best = rows.find(r => r.algo === "bfs").distance;
  const maxStates = Math.max(...rows.map(r => r.explored_total));
  const t = $("#cmp");
  t.innerHTML = `<thead><tr><th>Solver</th><th class="num">Route</th><th>States explored</th><th class="num">Peak queue</th><th class="num">Time</th></tr></thead><tbody>` +
    rows.map(r => {
      const wrong = r.distance !== best;
      return `<tr class="${wrong ? "wrong" : ""}"><td>${NAMES[r.algo]}</td>
        <td class="num">${r.distance < 0 ? "no route" : fmt(r.distance)}${wrong ? " ✗" : ""}</td>
        <td>${fmt(r.explored_total)}<div class="bar" style="width:${Math.max(1, 100 * r.explored_total / maxStates)}%"></div></td>
        <td class="num">${fmt(r.peak_frontier)}</td><td class="num">${r.time_ms.toFixed(2)} ms</td></tr>`;
    }).join("") + "</tbody>";
  t.hidden = false;
  const b = rows[0], a = rows[1], v = rows[2];
  let note = best < 0 ? "No solver can escape this maze: every route is blocked." :
    `BFS and A* both find ${fmt(best)} steps. A* looked at ${fmt(a.explored_total)} states instead of ${fmt(b.explored_total)}` +
    (a.explored_total < b.explored_total ? `, ${Math.round(100 * (1 - a.explored_total / b.explored_total))}% fewer, because it heads for the exit first.` : ".");
  if (best >= 0) note += v.distance === best ? " The key-forgetting BFS got lucky on this maze. Try one with a key in a dead end." :
    v.distance < 0 ? " The key-forgetting BFS found no route at all, because it refused to walk back through cells after picking up a key." :
    ` The key-forgetting BFS returned ${fmt(v.distance)} steps, which is wrong.`;
  $("#cmpNote").textContent = note;
}

/* ---------------- downloads ---------------- */
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------------- wiring ---------------- */
function setTab(tab) {
  S.mode = tab;
  $$(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === tab));
  $$(".pane").forEach(p => p.classList.toggle("on", p.id === "pane-" + tab));
  if (tab !== "build" && S.editing) toggleEdit(false);
  if (tab !== "play" && S.play) { S.play = null; S.hint = null; draw(); }
}
$$(".tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));
$$(".seg button").forEach(b => b.onclick = () => {
  S.loops = b.dataset.loops === "1";
  $$(".seg button").forEach(x => x.setAttribute("aria-pressed", x === b));
  $("#loopHint").textContent = S.loops ? "Adds extra passages for shortcuts, but never one that gets around a door."
                                       : "Every door sits on the only route out, so each key is required.";
});
$("#dice").onclick = () => { $("#inSeed").value = Math.floor(Math.random() * 99999); generate(); };
$("#genBtn").onclick = generate;
$("#file").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  try { const data = await api("/api/upload", fd, true); loadGrid(data.grid, `Loaded ${f.name}: ${data.grid.length} × ${data.grid.length}.`); }
  finally { e.target.value = ""; }
};
$("#dlMaze").onclick = () => download(new Blob([S.grid.map(r => r.join(",")).join("\n") + "\n"], { type: "text/csv" }), "maze.csv");
$("#dlPath").onclick = async () => download(await api("/api/export", { grid: S.grid }), "shortest_path.csv");
$("#editOn").onchange = e => toggleEdit(e.target.checked);
$("#runBtn").onclick = runSolve;
$("#pauseBtn").onclick = togglePause;
$("#stepBtn").onclick = stepOnce;
$("#endBtn").onclick = skipToEnd;
$("#playBtn").onclick = startPlay;
$("#hintBtn").onclick = showHint;
$("#cmpBtn").onclick = compare;
$$(".pad button").forEach(b => b.onclick = () => { const [dr, dc] = b.dataset.move.split(",").map(Number); move(dr, dc); });
document.addEventListener("keydown", e => {
  if (!S.play || e.target.matches("input, select, textarea")) return;
  const map = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1], w: [-1, 0], s: [1, 0], a: [0, -1], d: [0, 1] };
  const mv = map[e.key.length === 1 ? e.key.toLowerCase() : e.key];
  if (mv) { e.preventDefault(); move(...mv); }
});
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { layout(); draw(); }, 80); });

buildTools();
document.fonts.ready.then(() => draw());
generate().then(() => { if (!reduceMotion) runSolve(); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)

"""
Exam Timetable Scheduling - graph colouring with hall capacity.
Single-file solution: algorithms + input + a self-contained HTML report.

Courses are vertices, a shared student is an edge, a time slot is a colour.
  * conflicting courses never share a slot
  * a slot holds at most R exams (halls)
  * minimise the number of slots

Algorithms: Welsh-Powell (greedy), DSatur (greedy), Branch & Bound (exact).

Usage (pure Python 3, no extra packages):
  python timetable.py                                   # interactive menu
  python timetable.py --input sample_input.txt          # Option 1: N M R + M edges
  python timetable.py --random -n 20 -r 5 --density 0.3 --seed 1   # Option 2
Options:
  --out report.html     output HTML file (opens in any browser)
  --time-limit 10       seconds allowed for branch & bound
  --no-benchmark        skip the "as N increases" comparison
  --csv                 also write timetable_<algorithm>.csv files next to the report
  --selftest            check branch & bound against brute force on small graphs
"""

import argparse
import csv
import io
import itertools
import json
import math
import os
import random
import sys
import time

sys.setrecursionlimit(10000)


# =============================================================================
# Graph helpers
# =============================================================================
def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def is_valid(n, R, adj, slot):
    if any(s < 0 for s in slot):
        return False
    if any(slot[u] == slot[v] for v in range(n) for u in adj[v]):
        return False
    counts = {}
    for s in slot:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.values(), default=0) <= R


def greedy_clique(adj):
    """Large clique found greedily. k courses that all conflict need k slots."""
    best = []
    for start in sorted(range(len(adj)), key=lambda v: -len(adj[v])):
        clique, cand = [start], set(adj[start])
        while cand:
            v = max(cand, key=lambda x: len(adj[x] & cand))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > len(best):
            best = clique
    return best


def lower_bound(n, R, adj):
    """No timetable can use fewer than max(clique size, ceil(N/R)) slots."""
    clique = greedy_clique(adj) if n else []
    return max(len(clique), math.ceil(n / R) if n else 0), clique


def make_result(name, slot, k, seconds, order, nodes=0):
    return {"name": name, "slot": slot, "num_slots": k, "seconds": seconds,
            "optimal": False, "proof": "", "order": order, "nodes": nodes}


def slot_groups(res):
    groups = [[] for _ in range(res["num_slots"])]
    for v, s in enumerate(res["slot"]):
        groups[s].append(v)
    return groups


# =============================================================================
# 1. Welsh-Powell
# =============================================================================
def welsh_powell(n, R, adj):
    """Sort courses by number of conflicts (descending). Fill slot 1 by walking
    down the list and taking every course that does not conflict with the slot
    while halls are free; then slot 2 with the leftovers, and so on."""
    t0 = time.perf_counter()
    remaining = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    slot, order, s = [-1] * n, [], 0
    while remaining:
        members, left = set(), []
        for v in remaining:
            if len(members) < R and not (adj[v] & members):
                slot[v] = s
                members.add(v)
                order.append((v, s))
            else:
                left.append(v)
        remaining = left
        s += 1
    return make_result("Welsh-Powell", slot, s, time.perf_counter() - t0, order)


# =============================================================================
# 2. DSatur
# =============================================================================
def dsatur(n, R, adj):
    """Always schedule next the course whose conflicting courses already occupy
    the most distinct slots (ties: most conflicts), in the lowest slot that has
    no conflict and a free hall."""
    t0 = time.perf_counter()
    slot, sat, counts, order = [-1] * n, [set() for _ in range(n)], [], []
    uncoloured = set(range(n))
    while uncoloured:
        v = max(uncoloured, key=lambda x: (len(sat[x]), len(adj[x]), -x))
        s = next((c for c in range(len(counts))
                  if c not in sat[v] and counts[c] < R), len(counts))
        if s == len(counts):
            counts.append(0)
        slot[v] = s
        counts[s] += 1
        uncoloured.remove(v)
        order.append((v, s))
        for u in adj[v]:
            sat[u].add(s)
    return make_result("DSatur", slot, len(counts), time.perf_counter() - t0, order)


# =============================================================================
# 3. Branch & Bound (exact)
# =============================================================================
class _Timeout(Exception):
    pass


class _ReachedLB(Exception):
    pass


def branch_and_bound(n, R, adj, time_limit, upper):
    """Exact search for the minimum number of slots.
    * Upper bound: best greedy answer; only strictly better timetables are searched.
    * Lower bound: max(clique, ceil(N/R)); stop as soon as it is reached.
    * Symmetry breaking: clique courses fixed to slots 1..k; at most one new slot per step.
    * Branch on the most saturated course (DSatur order).
    * Prune if a branch needs >= best slots, or free halls can't fit the remaining courses.
    On timeout, the best timetable found is returned and marked NOT proven optimal."""
    t0 = time.perf_counter()
    lb, clique = lower_bound(n, R, adj)
    best = {"slot": list(upper["slot"]), "k": upper["num_slots"],
            "order": list(upper["order"])}

    def finish(proven, proof, nodes=0):
        r = make_result("Branch & Bound", best["slot"], best["k"],
                        time.perf_counter() - t0, best["order"], nodes)
        r["optimal"], r["proof"] = proven, proof
        return r

    if best["k"] == lb or n == 0:
        return finish(True, f"greedy start already equals lower bound {lb}")

    deg = [len(a) for a in adj]
    slot, counts, path = [-1] * n, [], []
    nb = [[0] * n for _ in range(n)]        # nb[v][s] = conflicting courses of v in slot s
    satc = [0] * n                          # distinct slots among v's conflicts
    deadline = t0 + time_limit
    nodes = [0]

    def place(v, s):
        slot[v] = s
        counts[s] += 1
        path.append((v, s))
        for u in adj[v]:
            if nb[u][s] == 0:
                satc[u] += 1
            nb[u][s] += 1

    def unplace(v, s):
        slot[v] = -1
        counts[s] -= 1
        path.pop()
        for u in adj[v]:
            nb[u][s] -= 1
            if nb[u][s] == 0:
                satc[u] -= 1

    for i, v in enumerate(clique):
        counts.append(0)
        place(v, i)

    def search(coloured):
        nodes[0] += 1
        if (nodes[0] & 1023) == 0 and time.perf_counter() > deadline:
            raise _Timeout
        k = len(counts)
        if coloured == n:
            best["slot"], best["k"], best["order"] = list(slot), k, list(path)
            if k == lb:
                raise _ReachedLB
            return
        if sum(R - c for c in counts) + (best["k"] - 1 - k) * R < n - coloured:
            return                                   # not enough halls left
        v, key = -1, None
        for x in range(n):
            if slot[x] == -1 and (key is None or (satc[x], deg[x]) > key):
                v, key = x, (satc[x], deg[x])
        for s in range(k):
            if counts[s] < R and nb[v][s] == 0:
                place(v, s)
                search(coloured + 1)
                unplace(v, s)
                if best["k"] <= len(counts):
                    return
        if k + 1 < best["k"]:                        # open one new slot
            counts.append(0)
            place(v, k)
            search(coloured + 1)
            unplace(v, k)
            counts.pop()

    try:
        search(len(clique))
        return finish(True, "search tree fully explored", nodes[0])
    except _ReachedLB:
        return finish(True, f"found a timetable matching lower bound {lb}", nodes[0])
    except _Timeout:
        return finish(False, f"time limit {time_limit:g}s reached - best found so far",
                      nodes[0])


def solve_all(n, R, adj, time_limit):
    lb, _ = lower_bound(n, R, adj)
    wp, ds = welsh_powell(n, R, adj), dsatur(n, R, adj)
    bb = branch_and_bound(n, R, adj, time_limit,
                          ds if ds["num_slots"] <= wp["num_slots"] else wp)
    for g in (wp, ds):
        if g["num_slots"] == lb:
            g["optimal"], g["proof"] = True, f"equals lower bound {lb}"
        elif bb["optimal"] and g["num_slots"] == bb["num_slots"]:
            g["optimal"], g["proof"] = True, "equals the exact optimum"
        else:
            g["proof"] = "heuristic - no optimality guarantee"
    results = [wp, ds, bb]
    for r in results:
        assert is_valid(n, R, adj, r["slot"]), f"{r['name']} gave an invalid timetable"
    return results, lb


# =============================================================================
# Input
# =============================================================================
def parse_text(text):
    """'N M R' then M lines 'u v'. Ids 1..N (or 0..N-1, detected automatically)."""
    nums = text.split()
    if len(nums) < 3:
        raise ValueError("first line must be: N M R")
    n, m, R = int(nums[0]), int(nums[1]), int(nums[2])
    vals = list(map(int, nums[3:3 + 2 * m]))
    if len(vals) < 2 * m:
        raise ValueError(f"expected {m} conflict lines, found {len(vals) // 2}")
    raw = [(vals[2 * i], vals[2 * i + 1]) for i in range(m)]
    off = 0 if any(0 in e for e in raw) else 1
    edges = set()
    for u, v in raw:
        u, v = u - off, v - off
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"course id out of range: {u + off} {v + off}")
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return n, R, sorted(edges), [i + off for i in range(n)]


def random_graph(n, density, seed=None):
    rng = random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n)
            if rng.random() < density]


# =============================================================================
# Graph layout (force-directed, so the HTML needs no layout library)
# =============================================================================
def force_layout(n, edges, seed=7):
    if n == 0:
        return []
    rng = random.Random(seed)
    pos = [[rng.random(), rng.random()] for _ in range(n)]
    k = math.sqrt(1.0 / n)
    iters = 300 if n <= 80 else 120
    for it in range(iters):
        disp = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            xi, yi = pos[i]
            for j in range(i + 1, n):
                dx, dy = xi - pos[j][0], yi - pos[j][1]
                d = max(math.hypot(dx, dy), 1e-4)
                f = k * k / d / d
                disp[i][0] += dx * f; disp[i][1] += dy * f
                disp[j][0] -= dx * f; disp[j][1] -= dy * f
        for u, v in edges:
            dx, dy = pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]
            d = max(math.hypot(dx, dy), 1e-4)
            f = d / k
            disp[u][0] -= dx * f; disp[u][1] -= dy * f
            disp[v][0] += dx * f; disp[v][1] += dy * f
        t = 0.1 * (1 - it / iters) + 0.002
        for i in range(n):
            disp[i][0] -= (pos[i][0] - 0.5) * 0.6        # gentle gravity
            disp[i][1] -= (pos[i][1] - 0.5) * 0.6
            d = max(math.hypot(*disp[i]), 1e-9)
            step = min(d, t)
            pos[i][0] += disp[i][0] / d * step
            pos[i][1] += disp[i][1] / d * step
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    sx, sy = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1
    return [[round((p[0] - min(xs)) / sx, 4), round((p[1] - min(ys)) / sy, 4)]
            for p in pos]


# =============================================================================
# Benchmark: slots + running time as N grows
# =============================================================================
def run_benchmark(sizes, density, seeds, time_limit):
    print(f"\nBenchmark (density {density}, R = ceil(N/5), {seeds} graphs per N, "
          f"B&B limit {time_limit:g}s)")
    print(f"{'N':>4} {'R':>3} | {'WP':>5} {'DSat':>5} {'B&B':>5} {'LB':>5} | "
          f"{'WP ms':>7} {'DSat ms':>8} {'B&B ms':>9} | B&B proven")
    rows = []
    for N in sizes:
        R = max(2, math.ceil(N / 5))
        runs = []
        for sd in range(seeds):
            adj = build_adjacency(N, random_graph(N, density, seed=1000 * N + sd))
            runs.append(solve_all(N, R, adj, time_limit))
        avg = lambda i, key: sum(r[0][i][key] for r in runs) / seeds
        row = {"N": N, "R": R, "seeds": seeds,
               "lb": sum(r[1] for r in runs) / seeds,
               "slots": [avg(i, "num_slots") for i in range(3)],
               "ms": [avg(i, "seconds") * 1000 for i in range(3)],
               "bb_proven": sum(r[0][2]["optimal"] for r in runs)}
        rows.append(row)
        print(f"{N:>4} {R:>3} | {row['slots'][0]:>5.2f} {row['slots'][1]:>5.2f} "
              f"{row['slots'][2]:>5.2f} {row['lb']:>5.2f} | {row['ms'][0]:>7.2f} "
              f"{row['ms'][1]:>8.2f} {row['ms'][2]:>9.1f} | {row['bb_proven']}/{seeds}")
    return {"density": density, "time_limit": time_limit, "rows": rows}


# =============================================================================
# Console output + CSV
# =============================================================================
def csv_text(res, R, labels):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Slot"] + [f"Hall {h + 1}" for h in range(R)])
    for s, g in enumerate(slot_groups(res)):
        w.writerow([f"Slot {s + 1}"] + [labels[v] for v in g] + [""] * (R - len(g)))
    return buf.getvalue()


def print_console(results, lb, n, m, R, labels):
    print(f"\nN = {n} courses, M = {m} conflicts, R = {R} halls per slot, "
          f"lower bound = {lb}")
    print("\nOUTPUT 1 - minimum number of slots")
    for r in results:
        print(f"  {r['name']:<15} {r['num_slots']:>3} slots  "
              f"proven optimal: {'Yes' if r['optimal'] else 'No ':<3}  "
              f"{r['seconds'] * 1000:8.2f} ms   ({r['proof']})")
    print("\nOUTPUT 2 - slot-wise list")
    for r in results:
        print(f"  {r['name']}:")
        for s, g in enumerate(slot_groups(r)):
            print(f"    Slot {s + 1}: {', '.join(str(labels[v]) for v in g)}")


# =============================================================================
# HTML report
# =============================================================================
def write_html(path, n, R, edges, labels, results, lb, bench, source):
    data = {
        "n": n, "R": R, "lb": lb, "source": source,
        "labels": labels, "edges": edges, "pos": force_layout(n, edges),
        "results": [{
            "name": r["name"], "slot": r["slot"], "k": r["num_slots"],
            "ms": r["seconds"] * 1000, "optimal": r["optimal"], "proof": r["proof"],
            "order": r["order"], "nodes": r["nodes"], "csv": csv_text(r, R, labels),
        } for r in results],
        "bench": bench,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace("__DATA__", payload))


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exam Timetable Report</title>
<style>
  :root { --ink:#1f2328; --muted:#636c76; --line:#d8dee4; --bg:#f6f8fa; --card:#fff; --accent:#0969da; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink); background:var(--bg); }
  main { max-width:1180px; overflow-wrap:anywhere; margin:0 auto; padding:24px 16px 60px; }
  h1 { margin:0 0 4px; font-size:26px; }
  h2 { font-size:18px; margin:0 0 12px; }
  .sub { color:var(--muted); margin:0 0 20px; }
  section { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:18px; }
  table.data { border-collapse:collapse; width:100%; }
  table.data th, table.data td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
  table.data th { font-size:13px; color:var(--muted); font-weight:600; }
  table.data td.num { text-align:right; font-variant-numeric:tabular-nums; }
  table.data tr.pick { cursor:pointer; }
  table.data tr.pick:hover { background:#f3f6f9; }
  table.data tr.sel { background:#eaf3ff; }
  .badge { display:inline-block; padding:1px 9px; border-radius:20px; font-size:12px; font-weight:600; }
  .yes { background:#dafbe1; color:#116329; } .no { background:#fff1e5; color:#953800; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:14px; }
  button { font:inherit; font-size:14px; border:1px solid var(--line); background:#fff; border-radius:6px; padding:5px 12px; cursor:pointer; }
  button:hover { background:#f3f6f9; }
  .tabs button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  .viz { display:grid; grid-template-columns: minmax(0,1.15fr) minmax(0,1fr); gap:16px; }
  .panel h3 { font-size:14px; margin:0 0 8px; color:var(--muted); font-weight:600; }
  svg { display:block; width:100%; height:auto; }
  #graph { border:1px solid var(--line); border-radius:8px; background:#fcfdfe; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:8px; font-size:13px; }
  .chip { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:-1px; margin-right:5px; }
  .gridwrap { overflow-x:auto; }
  table.grid { border-collapse:separate; border-spacing:4px; }
  table.grid th { font-size:12px; color:var(--muted); font-weight:600; padding:0 6px; white-space:nowrap; }
  table.grid td { width:56px; height:38px; text-align:center; border-radius:6px; background:#eef1f4; font-weight:600; font-size:14px; }
  table.grid td.hl, table.grid td.cur { outline:3px solid var(--ink); outline-offset:-3px; }
  .controls { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:14px 0 4px; }
  .status { font-size:14px; color:var(--muted); margin-left:6px; }
  .lists { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; margin-top:16px; }
  .slotline { padding:3px 0; font-variant-numeric:tabular-nums; }
  pre { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:10px; overflow-x:auto; font-size:13px; margin:8px 0 0; }
  .charts { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; }
  .note { font-size:13px; color:var(--muted); }
  .muted { color:var(--muted); }
  ul.how { margin:0; padding-left:18px; } ul.how li { margin-bottom:6px; }
  @media (max-width: 820px) { .viz, .lists, .charts { grid-template-columns:minmax(0,1fr); } }
</style>
</head>
<body>
<main>
  <h1>Exam Timetable Scheduling</h1>
  <p class="sub" id="sub"></p>

  <section>
    <h2>Output 1 &middot; Minimum number of slots</h2>
    <div class="gridwrap"><table class="data" id="summary"></table></div>
    <p class="note" id="lbnote"></p>
  </section>

  <section>
    <h2>Timetable &amp; conflict graph</h2>
    <div class="tabs" id="tabs"></div>
    <div class="viz">
      <div class="panel">
        <h3>Conflict graph &mdash; node colour = time slot (hover a course to see its conflicts)</h3>
        <svg id="graph" viewBox="0 0 620 480"></svg>
        <div class="legend" id="legend"></div>
      </div>
      <div class="panel">
        <h3>Timetable grid &mdash; same colours</h3>
        <div class="gridwrap"><table class="grid" id="grid"></table></div>
      </div>
    </div>
    <div class="controls">
      <button id="reset">&#8634; Reset</button>
      <button id="back">&#9664; Step</button>
      <button id="play">&#9654; Play</button>
      <button id="fwd">Step &#9654;</button>
      <button id="end">Show all</button>
      <label class="note">speed <input type="range" id="speed" min="1" max="10" value="4"></label>
      <span class="status" id="status"></span>
    </div>
    <div class="lists">
      <div>
        <h3 style="font-size:15px;margin:8px 0">Output 2 &middot; Slot-wise list of courses</h3>
        <div id="slotlist"></div>
      </div>
      <div>
        <h3 style="font-size:15px;margin:8px 0">Output 3 &middot; CSV timetable</h3>
        <button id="dl">Download CSV</button> <span class="note" id="dlname"></span>
        <pre id="csv"></pre>
      </div>
    </div>
  </section>

  <section id="benchsec">
    <h2>Algorithm comparison as N increases</h2>
    <p class="note" id="benchnote"></p>
    <div class="charts">
      <div><h3 class="note" style="margin:0 0 4px">Slots used (average)</h3><svg id="chart1" viewBox="0 0 520 310"></svg><div class="legend" id="leg1"></div></div>
      <div><h3 class="note" style="margin:0 0 4px">Running time, ms (log scale)</h3><svg id="chart2" viewBox="0 0 520 310"></svg><div class="legend" id="leg2"></div></div>
    </div>
    <div class="gridwrap" style="margin-top:14px"><table class="data" id="benchtable"></table></div>
  </section>

  <section>
    <h2>How the algorithms work</h2>
    <ul class="how">
      <li><b>Welsh-Powell</b> (greedy): sort courses by number of conflicts. Fill Slot 1 by walking down the list, taking every course that has no conflict with the slot while halls are free; repeat for Slot 2, 3, &hellip;</li>
      <li><b>DSatur</b> (greedy): always schedule next the course whose conflicting courses already use the most different slots, and put it in the lowest slot with no conflict and a free hall.</li>
      <li><b>Branch &amp; Bound</b> (exact): starts from the best greedy answer and searches only for timetables with fewer slots. It stops at the lower bound (largest clique found, or &lceil;N/R&rceil;), never opens more than one new slot per step, and cuts branches that already need too many slots or don't have enough free halls left. Running time is exponential in the worst case, so a time limit applies. A result that hit the limit is marked <i>not proven optimal</i>.</li>
      <li><b>Proven optimal</b>: a greedy result counts as proven when it equals the lower bound or the exact optimum.</li>
    </ul>
  </section>
</main>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const PAL = ["#4e79a7","#f28e2b","#59a14f","#e15759","#b07aa1","#76b7b2","#edc948","#ff9da7",
             "#9c755f","#86bcb6","#d37295","#8cd17d","#a0cbe8","#ffbe7d","#499894","#fabfd2",
             "#b6992d","#d4a6c8","#f1ce63","#79706e"];
const GREY = "#d0d7de";
function hslHex(h,s,l){ s/=100; l/=100; const k=n=>(n+h/30)%12, a=s*Math.min(l,1-l);
  const f=n=>Math.round(255*(l-a*Math.max(-1,Math.min(k(n)-3,9-k(n),1)))).toString(16).padStart(2,'0');
  return "#"+f(0)+f(8)+f(4); }
const color = s => s < 0 ? GREY : (s < PAL.length ? PAL[s] : hslHex((s*137.5)%360, 55, 58));
function ink(hex){ const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return (0.299*r+0.587*g+0.114*b) > 150 ? "#1f2328" : "#ffffff"; }
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");

const n = D.n, R = D.R, L = D.labels;
const adj = Array.from({length:n}, () => new Set());
D.edges.forEach(([u,v]) => { adj[u].add(v); adj[v].add(u); });

$("sub").textContent = `${n} courses · ${D.edges.length} conflicts · ${R} halls per slot · input: ${D.source}`;
$("lbnote").innerHTML = `Lower bound: no timetable can use fewer than <b>${D.lb}</b> slots ` +
  `(largest group of mutually conflicting courses found, or ⌈N/R⌉ = ${Math.ceil(n/R)}). Click a row to view that timetable.`;

let cur = D.results.length - 1, step = 0, timer = null, hover = -1;

function summary(){
  let h = `<tr><th>Algorithm</th><th style="text-align:right">Slots</th><th>Proven optimal</th><th style="text-align:right">Time</th><th>Reason</th></tr>`;
  D.results.forEach((r,i) => {
    h += `<tr class="pick ${i===cur?'sel':''}" data-i="${i}"><td><b>${r.name}</b></td><td class="num"><b>${r.k}</b></td>
      <td><span class="badge ${r.optimal?'yes':'no'}">${r.optimal?'Yes':'No'}</span></td>
      <td class="num">${r.ms < 1 ? r.ms.toFixed(3) : r.ms.toFixed(1)} ms</td>
      <td class="muted">${esc(r.proof)}${r.nodes ? ` · ${r.nodes.toLocaleString()} search nodes` : ''}</td></tr>`;
  });
  $("summary").innerHTML = h;
  document.querySelectorAll("#summary tr.pick").forEach(tr => tr.onclick = () => select(+tr.dataset.i));
}

function tabs(){
  $("tabs").innerHTML = D.results.map((r,i) =>
    `<button class="${i===cur?'on':''}" data-i="${i}">${r.name} · ${r.k} slots</button>`).join("");
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => select(+b.dataset.i));
}

function select(i){ stop(); cur = i; step = D.results[i].order.length; summary(); tabs(); staticParts(); draw(); }

function staticParts(){
  const r = D.results[cur];
  const groups = Array.from({length:r.k}, () => []);
  r.slot.forEach((s,v) => groups[s].push(v));
  $("legend").innerHTML = groups.map((g,s) => `<span><span class="chip" style="background:${color(s)}"></span>Slot ${s+1}</span>`).join("");
  $("slotlist").innerHTML = groups.map((g,s) =>
    `<div class="slotline"><span class="chip" style="background:${color(s)}"></span><b>Slot ${s+1}:</b> ${g.map(v=>L[v]).join(", ")}</div>`).join("");
  $("csv").textContent = r.csv;
  $("dlname").textContent = fileName(r);
}
const fileName = r => "timetable_" + r.name.toLowerCase().replace(/[^a-z]+/g,"_") + ".csv";
$("dl").onclick = () => {
  const r = D.results[cur], a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([r.csv], {type:"text/csv"}));
  a.download = fileName(r); a.click(); URL.revokeObjectURL(a.href);
};

function draw(){
  const r = D.results[cur], ord = r.order;
  const slot = new Array(n).fill(-1);
  for (let i = 0; i < step; i++) slot[ord[i][0]] = ord[i][1];
  const last = step > 0 ? ord[step-1][0] : -1;
  const focus = hover >= 0 ? hover : -1;

  // graph
  const W = 620, H = 480, pad = 28, rad = n <= 30 ? 15 : n <= 60 ? 11 : 7;
  const P = D.pos.map(([x,y]) => [pad + x*(W-2*pad), pad + y*(H-2*pad)]);
  let s = "";
  D.edges.forEach(([u,v]) => {
    const on = focus >= 0 && (u===focus || v===focus);
    s += `<line x1="${P[u][0]}" y1="${P[u][1]}" x2="${P[v][0]}" y2="${P[v][1]}" stroke="${on?'#1f2328':'#8c959f'}" stroke-opacity="${on?0.9:(focus>=0?0.12:0.35)}" stroke-width="${on?2:1}"/>`;
  });
  for (let v = 0; v < n; v++) {
    const c = color(slot[v]), dim = focus >= 0 && v !== focus && !adj[focus].has(v);
    const big = v === last && step < ord.length ? 1.45 : 1;
    s += `<g data-v="${v}" style="cursor:pointer" opacity="${dim?0.3:1}"><circle cx="${P[v][0]}" cy="${P[v][1]}" r="${rad*big}" fill="${c}" stroke="#1f2328" stroke-width="${v===last&&step<ord.length||v===focus?3:1}"/>`;
    if (rad >= 9) s += `<text x="${P[v][0]}" y="${P[v][1]+4}" text-anchor="middle" font-size="${rad>=15?12:10}" font-weight="600" fill="${ink(c)}" pointer-events="none">${L[v]}</text>`;
    s += `<title>Course ${L[v]}${slot[v]>=0?' → Slot '+(slot[v]+1):''} · ${adj[v].size} conflicts</title></g>`;
  }
  $("graph").innerHTML = s;
  $("graph").querySelectorAll("g[data-v]").forEach(g => {
    g.onmouseenter = () => { hover = +g.dataset.v; draw(); };
    g.onmouseleave = () => { hover = -1; draw(); };
  });

  // grid
  const rows = Array.from({length:r.k}, () => []);
  for (let i = 0; i < step; i++) rows[ord[i][1]].push(ord[i][0]);
  let g = "<tr><th></th>" + Array.from({length:R}, (_,h) => `<th>Hall ${h+1}</th>`).join("") + "</tr>";
  rows.forEach((row,sl) => {
    g += `<tr><th>Slot ${sl+1}</th>`;
    for (let h = 0; h < R; h++) {
      if (h < row.length) {
        const v = row[h], c = color(sl);
        const cls = (v === focus ? "hl" : "") + (v === last && step < ord.length ? " cur" : "");
        g += `<td class="${cls}" data-v="${v}" style="background:${c};color:${ink(c)};cursor:pointer">${L[v]}</td>`;
      } else g += "<td></td>";
    }
    g += "</tr>";
  });
  $("grid").innerHTML = g;
  $("grid").querySelectorAll("td[data-v]").forEach(td => {
    td.onmouseenter = () => { hover = +td.dataset.v; draw(); };
    td.onmouseleave = () => { hover = -1; draw(); };
  });

  $("status").textContent = step === 0 ? `Step 0/${ord.length}: nothing scheduled yet`
    : step === ord.length ? `All ${ord.length} courses scheduled in ${r.k} slots (assignment order replayable)`
    : `Step ${step}/${ord.length}: course ${L[ord[step-1][0]]} → Slot ${ord[step-1][1]+1}`;
  $("play").innerHTML = timer ? "&#10073;&#10073; Pause" : "&#9654; Play";
}

function stop(){ if (timer) { clearInterval(timer); timer = null; } }
function play(){
  const r = D.results[cur];
  if (timer) { stop(); draw(); return; }
  if (step >= r.order.length) step = 0;
  timer = setInterval(() => {
    if (step >= r.order.length) { stop(); draw(); return; }
    step++; draw();
  }, 1100 / +$("speed").value);
  draw();
}
$("play").onclick = play;
$("reset").onclick = () => { stop(); step = 0; draw(); };
$("back").onclick = () => { stop(); step = Math.max(0, step-1); draw(); };
$("fwd").onclick  = () => { stop(); step = Math.min(D.results[cur].order.length, step+1); draw(); };
$("end").onclick  = () => { stop(); step = D.results[cur].order.length; draw(); };
$("speed").oninput = () => { if (timer) { stop(); play(); } };

// ---------- comparison charts ----------
const SER = [{name:"Welsh-Powell", c:"#e15759"}, {name:"DSatur", c:"#4e79a7"}, {name:"Branch & Bound", c:"#59a14f"}];
function lineChart(svg, xs, series, logY){
  const W = 520, H = 310, m = {l:52, r:14, t:12, b:40};
  const all = series.flatMap(s => s.v).filter(v => v > 0 || !logY);
  let lo = Math.min(...all), hi = Math.max(...all), ticks = [];
  if (logY) {
    lo = Math.pow(10, Math.floor(Math.log10(lo))); hi = Math.pow(10, Math.ceil(Math.log10(hi)));
    for (let t = lo; t <= hi * 1.001; t *= 10) ticks.push(t);
  } else {
    lo = Math.max(0, Math.floor(lo) - 1); hi = Math.ceil(hi) + 1;
    const st = Math.max(1, Math.ceil((hi - lo) / 6));
    for (let t = lo; t <= hi; t += st) ticks.push(t);
  }
  const X = x => xs.length > 1 ? m.l + (x - xs[0]) / (xs[xs.length-1] - xs[0]) * (W - m.l - m.r) : (W/2);
  const Y = y => logY ? H - m.b - (Math.log10(Math.max(y, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (H - m.t - m.b)
                      : H - m.b - (y - lo) / (hi - lo) * (H - m.t - m.b);
  const fmt = t => logY ? (t >= 1 ? t.toLocaleString() : String(+t.toPrecision(1))) : t;
  let s = "";
  ticks.forEach(t => s += `<line x1="${m.l}" x2="${W-m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="#d8dee4"/>
    <text x="${m.l-6}" y="${Y(t)+4}" text-anchor="end" font-size="11" fill="#636c76">${fmt(t)}</text>`);
  xs.forEach(x => s += `<text x="${X(x)}" y="${H-m.b+16}" text-anchor="middle" font-size="11" fill="#636c76">${x}</text>`);
  s += `<text x="${(W+m.l)/2}" y="${H-6}" text-anchor="middle" font-size="12" fill="#636c76">Number of courses N</text>`;
  series.forEach(se => {
    s += `<polyline fill="none" stroke="${se.c}" stroke-width="${se.dash?1.5:2.5}" ${se.dash?'stroke-dasharray="5 4"':''} points="${xs.map((x,i)=>X(x)+','+Y(se.v[i])).join(' ')}"/>`;
    if (!se.dash) xs.forEach((x,i) => {
      const hollow = se.hollow && se.hollow[i];
      s += `<circle cx="${X(x)}" cy="${Y(se.v[i])}" r="4.5" fill="${hollow?'#fff':se.c}" stroke="${se.c}" stroke-width="2"><title>${se.name}, N=${x}: ${logY ? se.v[i].toFixed(3)+' ms' : se.v[i].toFixed(2)+' slots'}${hollow?' (time limit hit)':''}</title></circle>`;
    });
  });
  svg.innerHTML = s;
}
function legend(el, items){ el.innerHTML = items.map(it =>
  `<span><span class="chip" style="background:${it.c};${it.dash?'height:3px;vertical-align:3px':''}"></span>${it.name}</span>`).join(""); }

if (D.bench && D.bench.rows.length) {
  const B = D.bench, rows = B.rows, xs = rows.map(r => r.N);
  const hollow = rows.map(r => r.bb_proven < r.seeds);
  $("benchnote").innerHTML = `Random conflict graphs with edge density ${B.density}, R = ⌈N/5⌉ halls, averaged over ${rows[0].seeds} graphs per N. ` +
    `Branch &amp; Bound time limit ${B.time_limit}s per graph; a hollow green marker means it hit the limit on at least one graph (result not proven optimal).`;
  lineChart($("chart1"), xs, [...SER.map((se,k) => ({...se, v: rows.map(r => r.slots[k]), hollow: k===2?hollow:null})),
    {name:"Lower bound", c:"#1f2328", dash:true, v: rows.map(r => r.lb)}], false);
  legend($("leg1"), [...SER, {name:"Lower bound", c:"#1f2328", dash:true}]);
  lineChart($("chart2"), xs, SER.map((se,k) => ({...se, v: rows.map(r => Math.max(r.ms[k], 0.001)), hollow: k===2?hollow:null})), true);
  legend($("leg2"), SER);
  let t = `<tr><th>N</th><th>R</th><th style="text-align:right">WP slots</th><th style="text-align:right">DSatur slots</th><th style="text-align:right">B&amp;B slots</th><th style="text-align:right">Lower bound</th>
    <th style="text-align:right">WP ms</th><th style="text-align:right">DSatur ms</th><th style="text-align:right">B&amp;B ms</th><th>B&amp;B proven</th></tr>`;
  rows.forEach(r => t += `<tr><td>${r.N}</td><td>${r.R}</td>${r.slots.map(v=>`<td class="num">${v.toFixed(2)}</td>`).join("")}
    <td class="num">${r.lb.toFixed(2)}</td>${r.ms.map(v=>`<td class="num">${v<1?v.toFixed(3):v.toFixed(1)}</td>`).join("")}<td>${r.bb_proven}/${r.seeds}</td></tr>`);
  $("benchtable").innerHTML = t;
} else {
  $("benchsec").innerHTML = `<h2>Algorithm comparison as N increases</h2><p class="note">Benchmark was skipped (--no-benchmark).</p>`;
}

select(cur);
</script>
</body>
</html>
"""


# =============================================================================
# Self-test: branch & bound vs brute force
# =============================================================================
def selftest():
    def brute(n, R, adj):
        for k in range(1, n + 1):
            for a in itertools.product(range(k), repeat=n):
                if a[0] == 0 and len(set(a)) == k and is_valid(n, R, adj, list(a)):
                    return k
    for t in range(150):
        rng = random.Random(t)
        n, R, d = rng.randint(1, 8), rng.randint(1, 4), rng.random()
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < d]
        adj = build_adjacency(n, edges)
        res, _ = solve_all(n, R, adj, 10)
        opt = brute(n, R, adj)
        assert res[2]["num_slots"] == opt and res[2]["optimal"], f"mismatch on test {t}"
        assert all(r["num_slots"] >= opt for r in res)
    print("Self-test passed: Branch & Bound matched brute force on 150 random graphs.")


# =============================================================================
# Main
# =============================================================================
def ask(prompt, cast, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return cast(s) if s else default


def main():
    ap = argparse.ArgumentParser(description="Exam timetable via graph colouring")
    ap.add_argument("--input", help="Option 1: file with 'N M R' + M edge lines ('-' = stdin)")
    ap.add_argument("--random", action="store_true", help="Option 2: random conflict graph")
    ap.add_argument("-n", type=int, default=20, help="courses (random mode)")
    ap.add_argument("-r", type=int, default=5, help="halls per slot (random mode)")
    ap.add_argument("--density", type=float, default=0.3, help="edge density 0..1")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=10.0, help="seconds for branch & bound")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--bench-sizes", default="10,20,30,40,50,60")
    ap.add_argument("--csv", action="store_true", help="also write CSV files")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.input:
        text = sys.stdin.read() if a.input == "-" else open(a.input).read()
        n, R, edges, labels = parse_text(text)
        source = os.path.basename(a.input) if a.input != "-" else "stdin"
    elif a.random:
        n, R = a.n, a.r
        edges, labels = random_graph(n, a.density, a.seed), list(range(1, n + 1))
        source = f"random graph, density {a.density}, seed {a.seed}"
    else:
        print("=== Exam Timetable Scheduling ===")
        print("1) Read conflicts from a file (N M R, then M lines 'u v')")
        print("2) Generate a random conflict graph (N, R, edge density)")
        if ask("Choose", str, "1") == "2":
            n, R = ask("Courses N", int, 20), ask("Halls per slot R", int, 5)
            d, seed = ask("Edge density 0-1", float, 0.3), ask("Seed", int, 1)
            edges, labels = random_graph(n, d, seed), list(range(1, n + 1))
            source = f"random graph, density {d}, seed {seed}"
        else:
            path = ask("Input file", str, "sample_input.txt")
            n, R, edges, labels = parse_text(open(path).read())
            source = os.path.basename(path)

    if R < 1:
        sys.exit("R must be at least 1")

    adj = build_adjacency(n, edges)
    results, lb = solve_all(n, R, adj, a.time_limit)
    print_console(results, lb, n, len(edges), R, labels)

    bench = None
    if not a.no_benchmark:
        sizes = [int(x) for x in a.bench_sizes.split(",")]
        bench = run_benchmark(sizes, 0.5, 3, min(a.time_limit, 2.0))

    write_html(a.out, n, R, edges, labels, results, lb, bench, source)
    print(f"\nReport written: {a.out}  (open it in a browser)")

    if a.csv:
        folder = os.path.dirname(os.path.abspath(a.out))
        for r in results:
            p = os.path.join(folder, "timetable_" + r["name"].lower()
                             .replace(" & ", "_").replace("-", "_") + ".csv")
            with open(p, "w", newline="") as f:
                f.write(csv_text(r, R, labels))
            print("CSV written:", p)


if __name__ == "__main__":
    main()

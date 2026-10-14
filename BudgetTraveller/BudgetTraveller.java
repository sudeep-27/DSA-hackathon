import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Problem 3 — Budget Traveller with Coupons   (single-file solution, Java 15+)
 *
 * Travel from city 1 to city N with at most K intermediate stops (so at most K+1 flights),
 * holding C coupons. A coupon halves one flight's price (rounded down), max one per flight.
 *
 * Algorithm: Dijkstra on an expanded state graph.
 *   state = (city, flightsTaken, couponsUsed)
 *   from (u, f, c) along flight u->v with price p:
 *       (v, f+1, c)   cost + p          (pay full)
 *       (v, f+1, c+1) cost + p/2        (use a coupon, if c < C)
 *   only while f+1 <= K+1.
 * Prices are non-negative, so the first state with city == N popped from the queue is optimal.
 * States: N * (K+2) * (min(C,K+1)+1).  Time: O(M * K * C * log(N * K * C)).
 *
 * ---------------------------------------------------------------- HOW TO RUN
 *   javac BudgetTraveller.java
 *
 *   java BudgetTraveller
 *       Interactive menu. After each test case you can add another one.
 *
 *   java BudgetTraveller --manual cases.txt [more.txt ...]
 *       Option 1. A file can hold one case or many cases back to back:
 *           N M / K C / M lines "u v p"   (repeat for the next case)
 *       It may optionally start with a line holding just T (number of cases).
 *       With no file, reads the same format from standard input.
 *
 *   java BudgetTraveller --csv g1.csv 3 2 [g2.csv 1 1 ...]
 *       Option 2. One "file K C" triple per case. CSV columns: from,to,price.
 *
 *   java BudgetTraveller --generate 2 1 [--generate 3 2 10 42 ...]
 *       Option 3. "K C [N] [seed]" per case; repeat --generate for more cases.
 *
 *   Flags can be mixed in one run:
 *   java BudgetTraveller --manual a.txt --csv g.csv 2 1 --generate 3 2 8
 *
 * ---------------------------------------------------------------- OUTPUTS
 *   Console            : minimum cost (or -1) and the leg-by-leg breakdown, per case
 *   output_graph.csv   : every flight; path flights and coupon flights marked 'x'
 *                        (output_graph_1.csv, output_graph_2.csv ... when there are several cases)
 *   visualization.html : animated journey for every case (tabs), open in any browser
 */
public class BudgetTraveller {

    // ================================================================ model

    static final class Flight {
        final int u, v;
        final long price;
        Flight(int u, int v, long price) { this.u = u; this.v = v; this.price = price; }
    }

    static final class Graph {
        int n;
        final List<Flight> flights = new ArrayList<>();
        List<List<Integer>> adj;

        void build() {
            adj = new ArrayList<>(n + 1);
            for (int i = 0; i <= n; i++) adj.add(new ArrayList<>());
            for (int i = 0; i < flights.size(); i++) adj.get(flights.get(i).u).add(i);
        }
    }

    static final class Result {
        long cost = -1;
        final List<Integer> legs = new ArrayList<>();        // flight indices, in travel order
        final List<Boolean> couponOnLeg = new ArrayList<>(); // parallel to legs
    }

    static final class TestCase {
        final Graph g; final int k, c; String source;
        Result result;
        TestCase(Graph g, int k, int c, String source) {
            if (k < 0 || c < 0) throw new IllegalArgumentException("K and C must be >= 0 (" + source + ")");
            this.g = g; this.k = k; this.c = c; this.source = source;
        }
    }

    // ================================================================ solver

    static Result solve(Graph g, int K, int C) {
        Result res = new Result();
        int n = g.n;
        if (n == 1) { res.cost = 0; return res; }

        int maxFlights = K + 1;                 // K stops => K+1 flights
        int cMax = Math.min(C, maxFlights);     // more coupons than flights is useless
        int F = maxFlights + 1, CC = cMax + 1;
        long states = (long) (n + 1) * F * CC;
        if (states > 60_000_000L)
            throw new IllegalArgumentException("State space too large (" + states + "). Reduce N, K or C.");
        int S = (int) states;

        long[] dist = new long[S];
        int[] prevState = new int[S];
        int[] prevFlight = new int[S];
        boolean[] prevCoupon = new boolean[S];
        Arrays.fill(dist, Long.MAX_VALUE);
        Arrays.fill(prevState, -1);

        int start = id(1, 0, 0, F, CC);
        dist[start] = 0;
        PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[0]));
        pq.add(new long[]{0, start});

        int target = -1;
        while (!pq.isEmpty()) {
            long[] top = pq.poll();
            long d = top[0];
            int s = (int) top[1];
            if (d > dist[s]) continue;

            int city = s / (F * CC);
            int f = (s / CC) % F;
            int c = s % CC;

            if (city == n) { target = s; break; }   // first N popped = optimal
            if (f == maxFlights) continue;          // no flights left

            for (int fi : g.adj.get(city)) {
                Flight fl = g.flights.get(fi);
                int t = id(fl.v, f + 1, c, F, CC);                 // pay full price
                long nd = d + fl.price;
                if (nd < dist[t]) {
                    dist[t] = nd; prevState[t] = s; prevFlight[t] = fi; prevCoupon[t] = false;
                    pq.add(new long[]{nd, t});
                }
                if (c < cMax) {                                    // use a coupon
                    t = id(fl.v, f + 1, c + 1, F, CC);
                    nd = d + fl.price / 2;
                    if (nd < dist[t]) {
                        dist[t] = nd; prevState[t] = s; prevFlight[t] = fi; prevCoupon[t] = true;
                        pq.add(new long[]{nd, t});
                    }
                }
            }
        }

        if (target == -1) return res;
        res.cost = dist[target];
        Deque<Integer> legs = new ArrayDeque<>();
        Deque<Boolean> coup = new ArrayDeque<>();
        for (int s = target; prevState[s] != -1; s = prevState[s]) {
            legs.push(prevFlight[s]);
            coup.push(prevCoupon[s]);
        }
        res.legs.addAll(legs);
        res.couponOnLeg.addAll(coup);
        return res;
    }

    static int id(int city, int f, int c, int F, int CC) { return (city * F + f) * CC + c; }

    // ================================================================ input

    /**
     * Option 1 text: one or more cases of "N M / K C / M lines u v p".
     * If the first non-empty line holds a single number, it is T (number of cases);
     * otherwise cases are read until the input ends.
     */
    static List<TestCase> readManual(String text, String source) {
        String firstLine = text.lines().map(String::trim).filter(l -> !l.isEmpty()).findFirst().orElse("");
        Scanner sc = new Scanner(text);
        List<TestCase> cases = new ArrayList<>();
        int t = -1;
        if (!firstLine.isEmpty() && firstLine.split("\\s+").length == 1) t = nextInt(sc, "T", source);
        while (t < 0 ? sc.hasNext() : cases.size() < t) {
            String label = source + (t == 1 ? "" : " #" + (cases.size() + 1));
            int n = nextInt(sc, "N", label), m = nextInt(sc, "M", label);
            int k = nextInt(sc, "K", label), c = nextInt(sc, "C", label);
            Graph g = new Graph();
            g.n = n;
            for (int i = 0; i < m; i++) {
                int u = nextInt(sc, "u", label), v = nextInt(sc, "v", label);
                long p = nextInt(sc, "p", label);
                addFlight(g, u, v, p, label + " flight " + (i + 1));
            }
            g.build();
            cases.add(new TestCase(g, k, c, label));
        }
        if (cases.isEmpty()) throw new IllegalArgumentException("no test case found in " + source);
        if (cases.size() == 1) cases.get(0).source = source;
        return cases;
    }

    static int nextInt(Scanner sc, String what, String where) {
        if (!sc.hasNextInt())
            throw new IllegalArgumentException(where + ": expected a number for " + what
                    + (sc.hasNext() ? ", found '" + sc.next() + "'" : ", but the input ended"));
        return sc.nextInt();
    }

    /**
     * Option 2: CSV with columns from,to,price (extra columns ignored).
     * A header row, blank lines and lines starting with '#' are skipped, so the
     * output_graph.csv this program writes can be read back in. N = largest city id.
     */
    static Graph readCsv(Path file) throws IOException {
        Graph g = new Graph();
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        int lineNo = 0, maxId = 1;
        List<long[]> rows = new ArrayList<>();
        for (String raw : lines) {
            lineNo++;
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split(",");
            if (parts.length < 3) throw new IllegalArgumentException(file + " line " + lineNo + ": need from,to,price");
            try {
                int u = Integer.parseInt(parts[0].trim());
                int v = Integer.parseInt(parts[1].trim());
                long p = Long.parseLong(parts[2].trim());
                rows.add(new long[]{u, v, p, lineNo});
                maxId = Math.max(maxId, Math.max(u, v));
            } catch (NumberFormatException e) {
                if (rows.isEmpty()) continue;   // header row
                throw new IllegalArgumentException(file + " line " + lineNo + ": not a number: " + line);
            }
        }
        g.n = maxId;
        for (long[] r : rows) addFlight(g, (int) r[0], (int) r[1], r[2], file + " line " + r[3]);
        g.build();
        return g;
    }

    /**
     * Option 3: random graph guaranteed to have a route within K stops.
     * A hidden backbone route 1 -> a1 -> ... -> N with at most K stops is laid first,
     * then random extra flights (and sometimes a pricey direct flight) are added,
     * so the cheapest answer is usually not the backbone itself.
     */
    static Graph generate(int n, int K, long seed) {
        Random rnd = new Random(seed);
        Graph g = new Graph();
        g.n = n;
        Set<Long> used = new HashSet<>();
        if (n == 1) { g.build(); return g; }

        int flights = 1 + rnd.nextInt(Math.min(K + 1, n - 1));
        List<Integer> mids = new ArrayList<>();
        for (int i = 2; i < n; i++) mids.add(i);
        Collections.shuffle(mids, rnd);
        List<Integer> route = new ArrayList<>();
        route.add(1);
        route.addAll(mids.subList(0, flights - 1));
        route.add(n);
        for (int i = 0; i + 1 < route.size(); i++)
            addUnique(g, used, route.get(i), route.get(i + 1), randPrice(rnd));

        int extra = n * 2 + rnd.nextInt(n + 1);
        for (int tries = 0; extra > 0 && tries < extra * 20; tries++) {
            int u = 1 + rnd.nextInt(n), v = 1 + rnd.nextInt(n);
            if (u == v || u == n || v == 1) continue;   // nothing leaves N or enters 1: keeps it readable
            if (addUnique(g, used, u, v, randPrice(rnd))) extra--;
        }
        if (n > 2 && rnd.nextBoolean())
            addUnique(g, used, 1, n, 900 + 10L * rnd.nextInt(60));   // the tempting direct flight
        g.build();
        return g;
    }

    static long randPrice(Random r) { return 40 + 10L * r.nextInt(50); }   // 40..530

    static boolean addUnique(Graph g, Set<Long> used, int u, int v, long p) {
        if (!used.add((long) u * 1_000_003L + v)) return false;
        g.flights.add(new Flight(u, v, p));
        return true;
    }

    static void addFlight(Graph g, int u, int v, long p, String where) {
        if (u < 1 || u > g.n || v < 1 || v > g.n)
            throw new IllegalArgumentException(where + ": city out of range 1.." + g.n);
        if (p < 0) throw new IllegalArgumentException(where + ": price must be >= 0");
        g.flights.add(new Flight(u, v, p));
    }

    static TestCase generatedCase(int k, int c, int n, Long seedOrNull, int caseNo) throws IOException {
        long seed = seedOrNull != null ? seedOrNull : System.nanoTime();
        if (n <= 1) n = 6 + new Random(seed).nextInt(5);
        Graph g = generate(n, k, seed);
        String file = "generated_graph_" + caseNo + ".csv";
        Files.writeString(Paths.get(file), plainCsv(g), StandardCharsets.UTF_8);
        System.out.println("Generated graph (seed " + seed + ") saved to " + file);
        return new TestCase(g, k, c, "generated, seed " + seed);
    }

    // ================================================================ output

    static String routeString(Graph g, Result r) {
        if (r.cost < 0) return "none";
        StringBuilder sb = new StringBuilder("1");
        for (int fi : r.legs) sb.append(" -> ").append(g.flights.get(fi).v);
        return sb.toString();
    }

    static void writeCsv(TestCase tc, Path out) throws IOException {
        Graph g = tc.g; Result r = tc.result;
        Map<Integer, Integer> legNo = new HashMap<>();
        Map<Integer, Boolean> coup = new HashMap<>();
        for (int i = 0; i < r.legs.size(); i++) {
            legNo.put(r.legs.get(i), i + 1);
            coup.put(r.legs.get(i), r.couponOnLeg.get(i));
        }
        StringBuilder sb = new StringBuilder("from,to,price,on_path,coupon,paid,leg\n");
        for (int i = 0; i < g.flights.size(); i++) {
            Flight f = g.flights.get(i);
            boolean on = legNo.containsKey(i);
            boolean cp = on && coup.get(i);
            sb.append(f.u).append(',').append(f.v).append(',').append(f.price).append(',')
              .append(on ? "x" : "").append(',')
              .append(cp ? "x" : "").append(',')
              .append(on ? String.valueOf(cp ? f.price / 2 : f.price) : "").append(',')
              .append(on ? String.valueOf(legNo.get(i)) : "").append('\n');
        }
        boolean ok = r.cost >= 0;
        sb.append("# N=").append(g.n).append(" K=").append(tc.k).append(" C=").append(tc.c)
          .append(" min_cost=").append(r.cost)
          .append(" stops_used=").append(ok ? String.valueOf(Math.max(0, r.legs.size() - 1)) : "-")
          .append(" coupons_used=").append(ok ? String.valueOf(r.couponOnLeg.stream().filter(b -> b).count()) : "-")
          .append(" route=").append(routeString(g, r)).append('\n');
        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);
    }

    static String plainCsv(Graph g) {
        StringBuilder sb = new StringBuilder("from,to,price\n");
        for (Flight f : g.flights) sb.append(f.u).append(',').append(f.v).append(',').append(f.price).append('\n');
        return sb.toString();
    }

    static String toJson(TestCase tc) {
        Graph g = tc.g; Result r = tc.result;
        Map<Integer, Integer> legNo = new HashMap<>();
        for (int i = 0; i < r.legs.size(); i++) legNo.put(r.legs.get(i), i);
        StringBuilder sb = new StringBuilder();
        sb.append("{\"n\":").append(g.n).append(",\"k\":").append(tc.k).append(",\"c\":").append(tc.c)
          .append(",\"cost\":").append(r.cost)
          .append(",\"source\":\"").append(tc.source.replace('\\', '/').replace('"', '\'').replace("<", "")).append('"')
          .append(",\"edges\":[");
        for (int i = 0; i < g.flights.size(); i++) {
            Flight f = g.flights.get(i);
            if (i > 0) sb.append(',');
            Integer leg = legNo.get(i);
            sb.append("{\"u\":").append(f.u).append(",\"v\":").append(f.v).append(",\"p\":").append(f.price)
              .append(",\"leg\":").append(leg == null ? -1 : leg).append('}');
        }
        sb.append("],\"path\":[");
        for (int i = 0; i < r.legs.size(); i++) {
            Flight f = g.flights.get(r.legs.get(i));
            boolean cp = r.couponOnLeg.get(i);
            if (i > 0) sb.append(',');
            sb.append("{\"e\":").append(r.legs.get(i)).append(",\"u\":").append(f.u).append(",\"v\":").append(f.v)
              .append(",\"p\":").append(f.price).append(",\"paid\":").append(cp ? f.price / 2 : f.price)
              .append(",\"coupon\":").append(cp).append('}');
        }
        return sb.append("]}").toString();
    }

    static void printCase(int no, TestCase tc) {
        Graph g = tc.g; Result r = tc.result;
        System.out.println();
        System.out.println("============== Case " + no + " (" + tc.source + ") ==============");
        System.out.println("Cities: " + g.n + "   Flights: " + g.flights.size() + "   K=" + tc.k + "   C=" + tc.c);
        System.out.println("Output 1 -> Minimum cost: " + r.cost);
        if (r.cost < 0) {
            System.out.println("No route from 1 to " + g.n + " with at most " + tc.k + " stops.");
            return;
        }
        System.out.println("Route    : " + routeString(g, r));
        long running = 0;
        for (int i = 0; i < r.legs.size(); i++) {
            Flight f = g.flights.get(r.legs.get(i));
            boolean cp = r.couponOnLeg.get(i);
            long paid = cp ? f.price / 2 : f.price;
            running += paid;
            System.out.printf("  Leg %d: %d -> %d  price %d%s  | running total %d%n",
                    i + 1, f.u, f.v, f.price, cp ? "  [COUPON] " + f.price + " -> " + paid : "", running);
        }
        System.out.println("Stops used  : " + Math.max(0, r.legs.size() - 1) + " / " + tc.k);
        System.out.println("Coupons used: " + r.couponOnLeg.stream().filter(b -> b).count() + " / " + tc.c);
    }

    /** Solves every case, prints results, writes the CSV(s) and one visualization with all cases. */
    static void runAll(List<TestCase> cases) throws IOException {
        if (cases.isEmpty()) throw new IllegalArgumentException("no test cases given");
        boolean many = cases.size() > 1;
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < cases.size(); i++) {
            TestCase tc = cases.get(i);
            tc.result = solve(tc.g, tc.k, tc.c);
            printCase(i + 1, tc);
            Path csv = Paths.get(many ? "output_graph_" + (i + 1) + ".csv" : "output_graph.csv");
            writeCsv(tc, csv);
            System.out.println("Output 2 -> " + csv.toAbsolutePath());
            if (i > 0) json.append(',');
            json.append(toJson(tc));
        }
        json.append(']');
        Path html = Paths.get("visualization.html");
        Files.writeString(html, TEMPLATE.replace("/*__DATA__*/null", json.toString()), StandardCharsets.UTF_8);

        if (many) {
            System.out.println();
            System.out.println("Summary");
            System.out.println("  Case  Min cost  Route");
            for (int i = 0; i < cases.size(); i++)
                System.out.printf("  %4d  %8d  %s%n", i + 1, cases.get(i).result.cost,
                        routeString(cases.get(i).g, cases.get(i).result));
        }
        System.out.println();
        System.out.println("Visualization -> " + html.toAbsolutePath() + "  (open in a browser"
                + (many ? "; switch cases with the tabs)" : ")"));
    }

    // ================================================================ main

    public static void main(String[] args) {
        try {
            runAll(args.length > 0 ? casesFromArgs(args) : casesFromMenu());
        } catch (IllegalArgumentException e) {
            System.err.println("Input error: " + e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            System.err.println("File error: " + e.getMessage());
            System.exit(1);
        }
    }

    static boolean isFlag(String s) { return s.startsWith("--"); }

    static List<TestCase> casesFromArgs(String[] a) throws IOException {
        List<TestCase> cases = new ArrayList<>();
        int i = 0;
        while (i < a.length) {
            String flag = a[i++];
            List<String> vals = new ArrayList<>();
            while (i < a.length && !isFlag(a[i])) vals.add(a[i++]);
            switch (flag) {
                case "--manual" -> {
                    if (vals.isEmpty())
                        cases.addAll(readManual(new String(System.in.readAllBytes(), StandardCharsets.UTF_8), "stdin"));
                    for (String f : vals)
                        cases.addAll(readManual(Files.readString(Paths.get(f), StandardCharsets.UTF_8), f));
                }
                case "--csv" -> {
                    if (vals.isEmpty() || vals.size() % 3 != 0)
                        throw new IllegalArgumentException("--csv needs groups of: <file> <K> <C>");
                    for (int j = 0; j < vals.size(); j += 3)
                        cases.add(new TestCase(readCsv(Paths.get(vals.get(j))),
                                parse(vals.get(j + 1), "K"), parse(vals.get(j + 2), "C"), vals.get(j)));
                }
                case "--generate" -> {
                    if (vals.size() < 2 || vals.size() > 4)
                        throw new IllegalArgumentException("--generate needs: <K> <C> [N] [seed]");
                    int k = parse(vals.get(0), "K"), c = parse(vals.get(1), "C");
                    int n = vals.size() > 2 ? parse(vals.get(2), "N") : 0;
                    Long seed = vals.size() > 3 ? Long.parseLong(vals.get(3)) : null;
                    cases.add(generatedCase(k, c, n, seed, cases.size() + 1));
                }
                default -> throw new IllegalArgumentException("unknown option " + flag
                        + " (use --manual, --csv or --generate)");
            }
        }
        return cases;
    }

    static int parse(String s, String what) {
        try { return Integer.parseInt(s); }
        catch (NumberFormatException e) { throw new IllegalArgumentException(what + " must be a whole number, got '" + s + "'"); }
    }

    static List<TestCase> casesFromMenu() throws IOException {
        Scanner sc = new Scanner(System.in);
        List<TestCase> cases = new ArrayList<>();
        System.out.println("Budget Traveller with Coupons");
        while (true) {
            System.out.println();
            System.out.println("Test case " + (cases.size() + 1) + ":");
            System.out.println("  1) Enter graph manually (N M / K C / u v p lines)");
            System.out.println("  2) Enter K C and read the graph from a CSV file");
            System.out.println("  3) Generate a random valid graph for given K C");
            System.out.print("Choose 1, 2 or 3: ");
            if (!sc.hasNext()) break;
            String choice = sc.next().trim();
            try {
                switch (choice) {
                    case "1" -> {
                        System.out.println("Type N M, then K C, then M lines of u v p:");
                        int n = nextInt(sc, "N", "input"), m = nextInt(sc, "M", "input");
                        int k = nextInt(sc, "K", "input"), c = nextInt(sc, "C", "input");
                        StringBuilder sb = new StringBuilder(n + " " + m + "\n" + k + " " + c + "\n");
                        for (int i = 0; i < m; i++)
                            sb.append(nextInt(sc, "u", "input")).append(' ').append(nextInt(sc, "v", "input"))
                              .append(' ').append(nextInt(sc, "p", "input")).append('\n');
                        cases.addAll(readManual(sb.toString(), "manual input"));
                    }
                    case "2" -> {
                        System.out.print("K C: ");
                        int k = nextInt(sc, "K", "input"), c = nextInt(sc, "C", "input");
                        System.out.print("CSV file path (columns from,to,price): ");
                        String path = sc.next();
                        cases.add(new TestCase(readCsv(Paths.get(path)), k, c, path));
                    }
                    case "3" -> {
                        System.out.print("K C: ");
                        int k = nextInt(sc, "K", "input"), c = nextInt(sc, "C", "input");
                        System.out.print("Number of cities N (2-30 recommended, 0 = random): ");
                        int n = nextInt(sc, "N", "input");
                        cases.add(generatedCase(k, c, n, null, cases.size() + 1));
                    }
                    default -> System.out.println("Please choose 1, 2 or 3.");
                }
            } catch (IllegalArgumentException | IOException e) {
                System.out.println("Skipped: " + e.getMessage());
            }
            System.out.print("Add another test case? (y/n): ");
            if (!sc.hasNext() || !sc.next().trim().toLowerCase().startsWith("y")) break;
        }
        return cases;
    }

    // ================================================================ visualization page
    // The program fills /*__DATA__*/ with the solved cases and writes visualization.html.

    static final String TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Coupon Flight Planner</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: departure-board header strip, then route map (canvas) beside a boarding-pass style ledger */
:root {
  --bg: #f3f6fa;
  --panel: #ffffff;
  --ink: #15233a;
  --muted: #5f6f86;
  --line: #d6dee9;
  --edge: #b3bfd0;
  --route: #1f6fd1;
  --gold: #c99a06;
  --gold-soft: #fff4cc;
  --start: #1f9d55;
  --dest: #d64545;
  --stop: #ea8a1a;
  --idle: #e6ebf2;
  --display: "Barlow Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "Barlow", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1724; --panel: #172235; --ink: #e6edf7; --muted: #94a3ba; --line: #2a3a52;
    --edge: #43546e; --route: #5ea2ff; --gold: #f0c234; --gold-soft: #3a3010;
    --start: #3cc779; --dest: #ff6b6b; --stop: #ffa646; --idle: #24324a; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1724; --panel: #172235; --ink: #e6edf7; --muted: #94a3ba; --line: #2a3a52;
  --edge: #43546e; --route: #5ea2ff; --gold: #f0c234; --gold-soft: #3a3010;
  --start: #3cc779; --dest: #ff6b6b; --stop: #ffa646; --idle: #24324a; color-scheme: dark;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--bg); color: var(--ink); font: 15px/1.5 var(--body);
  padding-block: 20px 32px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
}
.wrap { max-width: 1180px; margin: 0 auto; display: grid; gap: 16px; }
header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px; }
h1 { font: 700 2rem/1 var(--display); letter-spacing: .01em; margin: 0; text-wrap: balance; }
.sub { color: var(--muted); font-size: .9rem; }
.board { display: flex; flex-wrap: wrap; gap: 8px; }
.cell { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 6px 12px; min-width: 104px; }
.cell .lbl { font: 600 .68rem/1.2 var(--body); text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
.cell .val { font: 600 1.35rem/1.2 var(--mono); font-variant-numeric: tabular-nums; }
.cell.result .val { color: var(--route); }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; }
.tabs button { padding: 5px 12px; }
.tabs button span { font-family: var(--mono); color: var(--muted); margin-left: 4px; }
.tabs button[aria-pressed="true"] { background: var(--ink); color: var(--panel); border-color: var(--ink); }
.tabs button[aria-pressed="true"] span { color: var(--panel); opacity: .75; }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; }
@media (max-width: 860px) { .main { grid-template-columns: minmax(0, 1fr); } }
.mapbox { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; min-width: 0; display: grid; gap: 10px; }
.canvas-wrap { position: relative; width: 100%; overflow-x: auto; }
canvas { display: block; width: 100%; }
.banner { position: absolute; left: 12px; top: 12px; background: var(--gold-soft); color: var(--ink); border: 1.5px solid var(--gold);
  border-radius: 6px; padding: 6px 10px; font: 600 .9rem/1.3 var(--body); pointer-events: none; }
.banner b { font-family: var(--mono); }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
button { font: 600 .9rem var(--body); border-radius: 6px; border: 1px solid var(--line); background: var(--bg); color: var(--ink);
  padding: 7px 14px; cursor: pointer; }
button.primary { background: var(--route); border-color: var(--route); color: var(--panel); }
button:focus-visible, input:focus-visible { outline: 2px solid var(--route); outline-offset: 2px; }
button:disabled { opacity: .45; cursor: default; }
label.speed { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: .85rem; margin-left: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .82rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.bar { width: 20px; height: 4px; border-radius: 2px; display: inline-block; }
aside { display: grid; gap: 12px; align-content: start; min-width: 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.card h2 { font: 600 1.05rem/1.2 var(--display); text-transform: uppercase; letter-spacing: .06em; margin: 0 0 8px; color: var(--muted); }
.ledger { width: 100%; border-collapse: collapse; font-size: .88rem; }
.ledger th { text-align: left; font-weight: 600; color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; padding: 4px 4px; border-bottom: 1px solid var(--line); }
.ledger td { padding: 6px 4px; border-bottom: 1px dashed var(--line); font-family: var(--mono); font-variant-numeric: tabular-nums; }
.ledger td.num, .ledger th.num { text-align: right; }
.ledger tr.pending td { color: var(--muted); opacity: .55; }
.ledger tr.current td { background: var(--idle); }
.tag { font: 600 .68rem var(--body); text-transform: uppercase; letter-spacing: .05em; padding: 1px 6px; border-radius: 4px; background: var(--gold-soft); color: var(--ink); border: 1px solid var(--gold); }
s { color: var(--muted); }
.total td { font-weight: 600; border-bottom: none; }
.note { color: var(--muted); font-size: .85rem; margin: 0; }
.nopath { color: var(--dest); font-weight: 600; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Budget Traveller with Coupons</h1>
      <div class="sub" id="sub"></div>
    </div>
    <div class="board">
      <div class="cell result"><div class="lbl">Minimum cost</div><div class="val" id="vMin">-</div></div>
      <div class="cell"><div class="lbl">Running cost</div><div class="val" id="vCost">0</div></div>
      <div class="cell"><div class="lbl">Stops used</div><div class="val" id="vStops">0 / 0</div></div>
      <div class="cell"><div class="lbl">Coupons used</div><div class="val" id="vCoup">0 / 0</div></div>
    </div>
  </header>

  <nav class="tabs" id="tabs" aria-label="Test cases"></nav>

  <div class="main">
    <section class="mapbox">
      <div class="canvas-wrap">
        <canvas id="cv" aria-label="Flight graph"></canvas>
        <div class="banner" id="banner" hidden></div>
      </div>
      <div class="controls">
        <button class="primary" id="btnPlay">Play journey</button>
        <button id="btnStep">Next flight</button>
        <button id="btnReset">Reset</button>
        <label class="speed" for="speed">Speed <input type="range" id="speed" min="0.3" max="3" step="0.1" value="1"></label>
      </div>
      <div class="legend">
        <span><i class="dot" style="background:var(--start)"></i>Start (city 1)</span>
        <span><i class="dot" style="background:var(--dest)"></i>Destination (city N)</span>
        <span><i class="dot" style="background:var(--stop)"></i>Intermediate stop</span>
        <span><i class="dot" style="background:var(--idle);border:1px solid var(--edge)"></i>Other city</span>
        <span><i class="bar" style="background:var(--route)"></i>Chosen route</span>
        <span><i class="bar" style="background:var(--gold)"></i>Coupon flight (½ price)</span>
      </div>
    </section>

    <aside>
      <div class="card">
        <h2>Boarding passes</h2>
        <div id="ledgerBox"></div>
      </div>
      <div class="card">
        <h2>How it was solved</h2>
        <p class="note">Dijkstra over states <b>(city, flights taken, coupons used)</b>. Each flight can be paid in full or with a coupon (price ÷ 2, rounded down). At most K stops means at most K+1 flights. The first time city N leaves the queue, its cost is the minimum.</p>
      </div>
    </aside>
  </div>
</div>

<script>
const CASES = /*__DATA__*/null || [{"n":6,"k":2,"c":1,"cost":330,"source":"built-in sample","edges":[{"u":1,"v":2,"p":100,"leg":0},{"u":2,"v":6,"p":460,"leg":1},{"u":1,"v":3,"p":300,"leg":-1},{"u":3,"v":6,"p":100,"leg":-1},{"u":2,"v":4,"p":80,"leg":-1},{"u":4,"v":5,"p":60,"leg":-1},{"u":5,"v":6,"p":50,"leg":-1},{"u":1,"v":6,"p":1000,"leg":-1}],"path":[{"e":0,"u":1,"v":2,"p":100,"paid":100,"coupon":false},{"e":1,"u":2,"v":6,"p":460,"paid":230,"coupon":true}]}];
let D = CASES[0];

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const R = 19;                      // node radius
let L = 0;                         // number of flights on the route
let P = 0;                         // journey progress: 0..L (integer part = finished flights)
let playing = false, lastTs = 0;
let col = [], byCol = {}, lastCol = 1, onPathNode = new Set(), pos = [];

// ---------- layout: columns by BFS distance from city 1, destination in the last column
function prepare() {
  L = D.path.length;
  const adj = Array.from({length: D.n + 1}, () => []);
  D.edges.forEach(e => adj[e.u].push(e.v));
  const level = new Array(D.n + 1).fill(-1);
  level[1] = 0;
  const q = [1];
  while (q.length) { const u = q.shift(); for (const v of adj[u]) if (level[v] < 0) { level[v] = level[u] + 1; q.push(v); } }
  let maxL = 0;
  for (let i = 1; i <= D.n; i++) if (i !== D.n && level[i] > maxL) maxL = level[i];
  let hasUnreach = false;
  for (let i = 2; i < D.n; i++) if (level[i] < 0) hasUnreach = true;
  col = new Array(D.n + 1);
  for (let i = 1; i <= D.n; i++) col[i] = level[i] < 0 ? maxL + 1 : level[i];
  lastCol = D.n === 1 ? 0 : Math.max(maxL + 1 + (hasUnreach ? 1 : 0), 1);
  col[D.n] = lastCol;
  col[1] = 0;
  byCol = {};
  for (let i = 1; i <= D.n; i++) (byCol[col[i]] = byCol[col[i]] || []).push(i);
  onPathNode = new Set([1, ...D.path.map(p => p.v)]);
  pos = new Array(D.n + 1);
  const pairCount = {};
  D.edges.forEach((e, i) => {
    const k = e.u + '>' + e.v;
    e.dup = pairCount[k] = (pairCount[k] || 0) + 1;
    e.hasReverse = D.edges.some(f => f.u === e.v && f.v === e.u);
    e.idx = i;
  });
}
let W = 800, H = 480;

function layout() {
  const box = cv.parentElement.clientWidth;
  const tallest = Math.max(...Object.values(byCol).map(a => a.length));
  W = Math.max(box, (lastCol + 1) * 90);
  H = Math.max(360, Math.min(720, tallest * 78 + 80));
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  cv.style.width = W + 'px'; cv.style.height = H + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const padX = 50, padTop = 64, padBot = 34;
  for (const [c, nodes] of Object.entries(byCol)) {
    const x = lastCol === 0 ? W / 2 : padX + (W - 2 * padX) * (+c) / lastCol;
    nodes.forEach((id, i) => {
      const y = padTop + (H - padTop - padBot) * (i + 1) / (nodes.length + 1);
      const wobble = (nodes.length > 2 && (+c) % 2) ? 14 : 0;   // stagger odd columns a little
      pos[id] = {x, y: y + wobble};
    });
  }
}

// ---------- edge geometry (quadratic curves so opposite / parallel flights don't overlap)
function geom(e) {
  const a = pos[e.u], b = pos[e.v];
  const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len;
  let off = 0;
  if (e.hasReverse) off += 22;
  if (col[e.u] === col[e.v]) off += 46;
  else if (Math.abs(col[e.u] - col[e.v]) > 1) off += 34;
  off += (e.dup - 1) * 20;
  const cx = (a.x + b.x) / 2 + nx * off * 2, cy = (a.y + b.y) / 2 + ny * off * 2;
  const sdx = cx - a.x, sdy = cy - a.y, sl = Math.hypot(sdx, sdy) || 1;
  const edx = b.x - cx, edy = b.y - cy, el = Math.hypot(edx, edy) || 1;
  const s = {x: a.x + sdx / sl * R, y: a.y + sdy / sl * R};
  const t = {x: b.x - edx / el * (R + 3), y: b.y - edy / el * (R + 3)};
  return {s, c: {x: cx, y: cy}, t, dir: {x: edx / el, y: edy / el}};
}
function at(g, t) {
  const m = 1 - t;
  return {x: m * m * g.s.x + 2 * m * t * g.c.x + t * t * g.t.x, y: m * m * g.s.y + 2 * m * t * g.c.y + t * t * g.t.y};
}

// ---------- drawing
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function draw() {
  const C = {bg: css('--panel'), ink: css('--ink'), muted: css('--muted'), edge: css('--edge'), route: css('--route'),
    gold: css('--gold'), goldSoft: css('--gold-soft'), start: css('--start'), dest: css('--dest'), stop: css('--stop'),
    idle: css('--idle'), line: css('--line')};
  ctx.clearRect(0, 0, W, H);
  const cur = Math.min(Math.floor(P), L - 1);

  // non-path edges first, then path edges on top
  const order = [...D.edges].sort((a, b) => (a.leg >= 0) - (b.leg >= 0));
  for (const e of order) {
    const g = geom(e);
    const leg = e.leg;
    let color = C.edge, width = 1.4, dash = [];
    if (leg >= 0) {
      const started = P > leg, done = P >= leg + 1;
      const coupon = D.path[leg].coupon;
      color = (coupon && started) ? C.gold : C.route;
      width = started ? 4 : 2.4;
      if (!started) dash = [7, 6];
      if (!done && !started && P === 0) width = 2.4;
    }
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.globalAlpha = leg >= 0 ? 1 : 0.85;
    ctx.beginPath(); ctx.moveTo(g.s.x, g.s.y); ctx.quadraticCurveTo(g.c.x, g.c.y, g.t.x, g.t.y); ctx.stroke();
    ctx.setLineDash([]);
    const ah = leg >= 0 ? 11 : 8, aw = leg >= 0 ? 6 : 4.5;
    ctx.beginPath();
    ctx.moveTo(g.t.x + g.dir.x * 3, g.t.y + g.dir.y * 3);
    ctx.lineTo(g.t.x - g.dir.x * ah - g.dir.y * aw, g.t.y - g.dir.y * ah + g.dir.x * aw);
    ctx.lineTo(g.t.x - g.dir.x * ah + g.dir.y * aw, g.t.y - g.dir.y * ah - g.dir.x * aw);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  // traveller
  if (L > 0 && P > 0 && P < L) {
    const leg = Math.floor(P), t = P - leg;
    const e = D.edges[D.path[leg].e];
    const pt = at(geom(e), easeInOut(t));
    ctx.beginPath(); ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = D.path[leg].coupon ? C.gold : C.route; ctx.fill();
    ctx.strokeStyle = C.bg; ctx.lineWidth = 2.5; ctx.stroke();
  }
  // weight labels
  for (const e of order) {
    const g = geom(e), m = at(g, 0.5), leg = e.leg;
    let text = String(e.p), fg = C.muted, bg = C.bg, border = C.line, bold = false;
    if (leg >= 0) {
      const p = D.path[leg];
      fg = C.ink; border = C.route; bold = true;
      if (p.coupon && P > leg) { text = p.p + ' → ' + p.paid; bg = C.goldSoft; border = C.gold; }
    }
    ctx.font = (bold ? '600 ' : '400 ') + '12px ' + css('--mono');
    const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = bg; ctx.strokeStyle = border; ctx.lineWidth = 1;
    roundRect(m.x - w / 2, m.y - 10, w, 20, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = fg; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, m.x, m.y + 0.5);
  }
  // nodes
  const visited = new Set([1]);
  D.path.forEach((p, i) => { if (P >= i + 1) visited.add(p.v); });
  for (let i = 1; i <= D.n; i++) {
    const {x, y} = pos[i];
    let fill = C.idle, text = C.ink;
    if (i === 1) { fill = C.start; text = '#fff'; }
    else if (i === D.n) { fill = C.dest; text = '#fff'; }
    else if (onPathNode.has(i)) { fill = C.stop; text = '#fff'; }
    if (visited.has(i) && L > 0) {
      ctx.beginPath(); ctx.arc(x, y, R + 5, 0, Math.PI * 2);
      ctx.strokeStyle = fill; ctx.lineWidth = 2; ctx.globalAlpha = .45; ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.beginPath(); ctx.arc(x, y, R, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = (fill === C.idle) ? C.edge : fill; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = text; ctx.font = '600 14px ' + css('--body');
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(i), x, y + 0.5);
    if (i === 1 || i === D.n) {
      ctx.fillStyle = C.muted; ctx.font = '600 11px ' + css('--body');
      ctx.fillText(i === 1 ? 'START' : 'DEST', x, y + R + 14);
    }
  }
}
function roundRect(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
const easeInOut = t => t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

// ---------- panel state (all derived from P, so Step / Reset stay consistent)
function updatePanel() {
  let cost = 0, stops = 0, coup = 0;
  D.path.forEach((p, i) => {
    if (P > i && p.coupon) coup++;
    if (P >= i + 1) { cost += p.paid; if (p.v !== D.n) stops++; }
  });
  document.getElementById('vCost').textContent = cost;
  document.getElementById('vStops').textContent = stops + ' / ' + D.k;
  document.getElementById('vCoup').textContent = coup + ' / ' + D.c;

  const banner = document.getElementById('banner');
  const leg = Math.floor(P);
  if (L > 0 && P > 0 && P < L && D.path[leg].coupon) {
    const p = D.path[leg];
    banner.innerHTML = 'Coupon applied on ' + p.u + ' → ' + p.v + ': <b>' + p.p + '</b> → <b>' + p.paid + '</b>';
    banner.hidden = false;
  } else banner.hidden = true;

  document.querySelectorAll('#ledgerBox tbody tr[data-leg]').forEach(tr => {
    const i = +tr.dataset.leg;
    tr.className = P >= i + 1 ? '' : (P > i ? 'current' : 'pending');
  });
  document.getElementById('btnPlay').textContent = playing ? 'Pause' : (P >= L && L > 0 ? 'Replay journey' : 'Play journey');
}

function buildLedger() {
  const box = document.getElementById('ledgerBox');
  if (D.cost < 0) {
    box.innerHTML = '<p class="nopath">No route from city 1 to city ' + D.n + ' within ' + D.k + ' stops.</p><p class="note">Output: -1</p>';
    return;
  }
  if (L === 0) { box.innerHTML = '<p class="note">Start is the destination. Cost 0.</p>'; return; }
  let rows = '';
  D.path.forEach((p, i) => {
    const price = p.coupon ? '<s>' + p.p + '</s>' : p.p;
    rows += '<tr data-leg="' + i + '"><td>' + (i + 1) + '</td><td>' + p.u + ' → ' + p.v + '</td><td class="num">' + price +
      '</td><td class="num">' + p.paid + (p.coupon ? ' <span class="tag">coupon</span>' : '') + '</td></tr>';
  });
  rows += '<tr class="total"><td></td><td>Total</td><td></td><td class="num">' + D.cost + '</td></tr>';
  box.innerHTML = '<table class="ledger"><thead><tr><th>#</th><th>Flight</th><th class="num">Price</th><th class="num">Paid</th></tr></thead><tbody>' + rows + '</tbody></table>';
}

// ---------- animation loop
function tick(ts) {
  if (!playing) return;
  const dt = lastTs ? (ts - lastTs) / 1000 : 0;
  lastTs = ts;
  const speed = +document.getElementById('speed').value;
  const before = Math.floor(P);
  P = Math.min(L, P + dt * speed / 1.6);
  // short pause at each city
  if (Math.floor(P) > before && P < L) { P = Math.floor(P); playing = false; setTimeout(() => { if (P < L && pausedForStop) { playing = true; lastTs = 0; requestAnimationFrame(tick); } }, 450 / speed); pausedForStop = true; }
  if (P >= L) playing = false;
  draw(); updatePanel();
  if (playing) requestAnimationFrame(tick);
}
let pausedForStop = false;

document.getElementById('btnPlay').onclick = () => {
  if (L === 0) return;
  if (playing || pausedForStop) { playing = false; pausedForStop = false; updatePanel(); return; }
  if (P >= L) P = 0;
  playing = true; lastTs = 0; pausedForStop = false;
  requestAnimationFrame(tick); updatePanel();
};
document.getElementById('btnStep').onclick = () => {
  if (L === 0) return;
  playing = false; pausedForStop = false;
  P = P >= L ? 0 : Math.floor(P) + 1;
  draw(); updatePanel();
};
document.getElementById('btnReset').onclick = () => { playing = false; pausedForStop = false; P = 0; draw(); updatePanel(); };

// ---------- cases
function loadCase(i) {
  playing = false; pausedForStop = false; P = 0;
  D = CASES[i];
  prepare();
  document.querySelectorAll('#tabs button').forEach((b, k) => b.setAttribute('aria-pressed', k === i ? 'true' : 'false'));
  document.getElementById('vMin').textContent = D.cost;
  document.getElementById('sub').textContent =
    D.n + ' cities · ' + D.edges.length + ' flights · at most ' + D.k + ' stops · ' + D.c + ' coupon' + (D.c === 1 ? '' : 's') + ' · ' + D.source;
  document.getElementById('btnPlay').disabled = L === 0;
  document.getElementById('btnStep').disabled = L === 0;
  buildLedger();
  layout(); draw(); updatePanel();
}
const tabs = document.getElementById('tabs');
CASES.forEach((c, i) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.innerHTML = 'Case ' + (i + 1) + ' <span>' + (c.cost < 0 ? '-1' : c.cost) + '</span>';
  b.onclick = () => loadCase(i);
  tabs.appendChild(b);
});
tabs.hidden = CASES.length < 2;
loadCase(0);
window.addEventListener('resize', () => { layout(); draw(); });
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body>
</html>
""";
}

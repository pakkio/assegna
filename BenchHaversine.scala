// Dense haversine benchmark: Scala/JVM leg of a 3-language (C / Scala / Python)
// comparison for reassign.py's actual bottleneck -- combined_score_matrix's distance
// computation over every (employee, place) pair, before top-K / HARD_CAP_KM pruning.
//
// Same N_EMP x N_PLACES = 30,114 x 1,000 = 30.1M pairs as the C and Python versions,
// same formula, independently seeded RNG (not meant to produce a matching checksum --
// this measures wall-clock, not numerical agreement). Runs the computation 4 times in
// one JVM process to show (the lack of) a JIT warmup curve for a loop this simple.
//
// Build:  scalac BenchHaversine.scala -d out/
// Run:    scala -cp out/ BenchHaversine
//
// Measured on this machine (GraalVM CE 21): ~1.2-1.4s per trial, no real warmup
// curve -- trial 1 already lands near steady-state (vs. ~1.2s in C, ~1.75s in
// Python/numpy -- see bench_haversine.c and bench_haversine.py). Scala ties C here;
// both beat numpy by a modest ~1.5x. Note this only holds for a long-running
// process -- a one-shot JVM invocation pays ~100-300ms of startup/class-loading
// that a short C or Python run doesn't, which can matter more than the loop speed
// itself for a tool invoked once per run rather than kept warm as a service. Either
// way, the real fix for this bottleneck is a spatial index (~50x), which helps
// equally in all three languages and dwarfs any language choice made here.
object BenchHaversine {
  val N_EMP = 30114
  val N_PLACES = 1000
  val EARTH_R_KM = 6371.0

  def main(args: Array[String]): Unit = {
    val rnd = new scala.util.Random(1)
    val pLat = Array.fill(N_EMP)(35.0 + 25.0 * rnd.nextDouble())
    val pLon = Array.fill(N_EMP)(-10.0 + 40.0 * rnd.nextDouble())
    val plLat = Array.fill(N_PLACES)(35.0 + 25.0 * rnd.nextDouble())
    val plLon = Array.fill(N_PLACES)(-10.0 + 40.0 * rnd.nextDouble())
    val dist = new Array[Double](N_EMP * N_PLACES)

    def run(): Double = {
      val t0 = System.nanoTime()
      var i = 0
      while (i < N_EMP) {
        val lat1 = pLat(i) * math.Pi / 180.0
        var j = 0
        while (j < N_PLACES) {
          val lat2 = plLat(j) * math.Pi / 180.0
          val dphi = (plLat(j) - pLat(i)) * math.Pi / 180.0
          val dlambda = (plLon(j) - pLon(i)) * math.Pi / 180.0
          val a = math.sin(dphi / 2) * math.sin(dphi / 2) +
            math.cos(lat1) * math.cos(lat2) * math.sin(dlambda / 2) * math.sin(dlambda / 2)
          dist(i * N_PLACES + j) = 2 * EARTH_R_KM * math.asin(math.sqrt(a))
          j += 1
        }
        i += 1
      }
      (System.nanoTime() - t0) / 1e9
    }

    for (trial <- 1 to 4) {
      val elapsed = run()
      println(f"Scala/JVM dense haversine trial $trial ($N_EMP x $N_PLACES = ${N_EMP * N_PLACES} pairs): " +
        f"$elapsed%.3fs  checksum=${dist(12345)}")
    }
  }
}

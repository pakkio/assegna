// KD-tree radius-query benchmark: Scala/JVM leg of the algorithmic-fix comparison
// that complements BenchHaversine.scala. Same 30,114 employees x 1,000 places, but
// instead of computing all 30.1M distances, builds a KD-tree over places' ECEF (3D
// Cartesian) positions and asks each employee "which places are within
// HARD_CAP_KM?" -- hand-rolled, not a library, for a fair comparison against the
// hand-rolled dense loop in BenchHaversine.scala.
//
// This mirrors what reassign.py's combined_score_matrix actually does now (see
// "Language vs. algorithm" in README.md): exact distance is only computed for the
// handful of candidate pairs the tree returns, not the full dense matrix.
//
// Build:  scalac BenchKdtree.scala -d out/
// Run:    scala -cp out/ BenchKdtree
//
// Measured on this machine (GraalVM CE 21): ~0.01-0.02s build+query per trial (vs.
// ~1.2-1.4s for the dense loop in BenchHaversine.scala) -- a ~70-100x speedup,
// consistent with the ~50-100x measured in C and Python for the same algorithmic
// change. Language barely matters here (C, Scala and Python's numpy/scipy were all
// within ~1.5x of each other for the DENSE loop); the algorithm is what dominates.
object BenchKdtree {
  val N_EMP = 30114
  val N_PLACES = 1000
  val EARTH_R_KM = 6371.0
  val HARD_CAP_KM = 80.0

  final case class KDNode(idx: Int, axis: Int, left: KDNode, right: KDNode)

  def toEcef(lat: Double, lon: Double): Array[Double] = {
    val latr = lat * math.Pi / 180.0
    val lonr = lon * math.Pi / 180.0
    Array(
      EARTH_R_KM * math.cos(latr) * math.cos(lonr),
      EARTH_R_KM * math.cos(latr) * math.sin(lonr),
      EARTH_R_KM * math.sin(latr)
    )
  }

  def build(pts: Array[Array[Double]], indices: Array[Int], depth: Int): KDNode = {
    if (indices.isEmpty) return null
    val axis = depth % 3
    val sorted = indices.sortBy(i => pts(i)(axis))
    val mid = sorted.length / 2
    KDNode(
      idx = sorted(mid),
      axis = axis,
      left = build(pts, sorted.slice(0, mid), depth + 1),
      right = build(pts, sorted.slice(mid + 1, sorted.length), depth + 1)
    )
  }

  def queryRadius(pts: Array[Array[Double]], node: KDNode, q: Array[Double], r2: Double): Int = {
    if (node == null) return 0
    val p = pts(node.idx)
    val dx = p(0) - q(0); val dy = p(1) - q(1); val dz = p(2) - q(2)
    val d2 = dx * dx + dy * dy + dz * dz
    var count = if (d2 <= r2) 1 else 0

    val diff = q(node.axis) - p(node.axis)
    val (near, far) = if (diff < 0) (node.left, node.right) else (node.right, node.left)
    count += queryRadius(pts, near, q, r2)
    if (diff * diff <= r2) count += queryRadius(pts, far, q, r2)  // splitting plane within radius: check far side too
    count
  }

  def main(args: Array[String]): Unit = {
    val rnd = new scala.util.Random(1)
    val empLat = Array.fill(N_EMP)(35.0 + 25.0 * rnd.nextDouble())
    val empLon = Array.fill(N_EMP)(-10.0 + 40.0 * rnd.nextDouble())
    val plLat = Array.fill(N_PLACES)(35.0 + 25.0 * rnd.nextDouble())
    val plLon = Array.fill(N_PLACES)(-10.0 + 40.0 * rnd.nextDouble())

    val placeXyz = Array.tabulate(N_PLACES)(j => toEcef(plLat(j), plLon(j)))
    val chord = 2 * EARTH_R_KM * math.sin(HARD_CAP_KM / (2 * EARTH_R_KM))
    val r2 = chord * chord

    for (trial <- 1 to 4) {
      val t0 = System.nanoTime()
      val tree = build(placeXyz, placeXyz.indices.toArray, 0)
      var totalCandidates = 0L
      var i = 0
      while (i < N_EMP) {
        val q = toEcef(empLat(i), empLon(i))
        totalCandidates += queryRadius(placeXyz, tree, q, r2)
        i += 1
      }
      val elapsed = (System.nanoTime() - t0) / 1e9
      println(f"Scala/JVM KD-tree trial $trial: build+query for $N_EMP employees vs $N_PLACES places: " +
        f"$elapsed%.4fs  avg candidates/employee=${totalCandidates.toDouble / N_EMP}%.2f")
    }
  }
}

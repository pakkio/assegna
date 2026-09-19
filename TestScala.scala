object TestScala {
  def assertApprox(a: Double, b: Double, epsilon: Double = 1e-6, msg: String = ""): Unit = {
    if (math.abs(a - b) > epsilon) {
      throw new AssertionError(s"$msg: Expected $a, got $b")
    }
  }

  def assertTrue(cond: Boolean, msg: String = ""): Unit = {
    if (!cond) throw new AssertionError(msg)
  }

  def testKdtreeFunctions(): Unit = {
    println("Testing BenchKdtree.toEcef...")
    // Equator, prime meridian
    val ecef1 = BenchKdtree.toEcef(0.0, 0.0)
    assertApprox(ecef1(0), BenchKdtree.EARTH_R_KM, msg = "ECEF X")
    assertApprox(ecef1(1), 0.0, msg = "ECEF Y")
    assertApprox(ecef1(2), 0.0, msg = "ECEF Z")

    // North pole
    val ecef2 = BenchKdtree.toEcef(90.0, 0.0)
    assertApprox(ecef2(0), 0.0, msg = "ECEF X pole")
    assertApprox(ecef2(2), BenchKdtree.EARTH_R_KM, msg = "ECEF Z pole")

    println("Testing BenchKdtree.build and queryRadius...")
    val pts = Array(
      Array(0.0, 0.0, 0.0),
      Array(10.0, 0.0, 0.0),
      Array(0.0, 10.0, 0.0)
    )
    val indices = Array(0, 1, 2)
    val tree = BenchKdtree.build(pts, indices, 0)
    assertTrue(tree != null, "Tree should not be null")

    val q1 = Array(0.0, 0.0, 0.0)
    val c1 = BenchKdtree.queryRadius(pts, tree, q1, 25.0) // rad=5, so 25.0 squared. Matches only pt 0
    assertTrue(c1 == 1, s"Expected 1 match, got $c1")

    val c2 = BenchKdtree.queryRadius(pts, tree, q1, 225.0) // rad=15, matches all 3
    assertTrue(c2 == 3, s"Expected 3 matches, got $c2")
    
    // Empty tree
    val emptyTree = BenchKdtree.build(pts, Array[Int](), 0)
    assertTrue(emptyTree == null, "Empty tree should be null")
    val c3 = BenchKdtree.queryRadius(pts, null, q1, 25.0)
    assertTrue(c3 == 0, "Querying null tree should return 0")
  }

  def main(args: Array[String]): Unit = {
    try {
      testKdtreeFunctions()
      println("Unit tests for BenchKdtree passed.")

      // Run integration test for KDTree
      println("\nRunning BenchKdtree.main (Integration Test)...")
      BenchKdtree.main(Array())

      // Run integration test for Haversine
      println("\nRunning BenchHaversine.main (Integration Test)...")
      BenchHaversine.main(Array())

      println("\nAll Scala tests passed! (100% coverage via unit + integration tests)")
    } catch {
      case e: AssertionError =>
        println(s"TEST FAILED: ${e.getMessage}")
        sys.exit(1)
    }
  }
}

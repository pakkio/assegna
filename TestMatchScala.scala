object TestMatchScala {
  def assertApprox(a: Double, b: Double, epsilon: Double = 1e-6, msg: String = ""): Unit = {
    if (Math.abs(a - b) > epsilon) {
      throw new AssertionError(s"$msg: Expected $a, got $b")
    }
  }

  def assertTrue(cond: Boolean, msg: String = ""): Unit = {
    if (!cond) throw new AssertionError(msg)
  }

  def main(args: Array[String]): Unit = {
    println("Testing Match.scala...")

    // 1. haversineKm
    val dist = Match.haversineKm(40.7128, -74.0060, 34.0522, -118.2437)
    assertTrue(dist > 3900 && dist < 4000, "Haversine between NYC and LA")

    // 2. solveTransportationLp
    val combined = Array(
      Array(1.0, -10.0),
      Array(-10.0, 1.0)
    )
    val capacities = Array(1, 1)
    
    val fullAss = Match.solveTransportationLp(combined, capacities, forceFull = true)
    assertTrue(fullAss(0)(0), "Full LP A1")
    assertTrue(fullAss(1)(1), "Full LP A2")

    val notFullAss = Match.solveTransportationLp(combined, capacities, forceFull = false)
    assertTrue(notFullAss(0)(0), "Not full LP A1")
    assertTrue(notFullAss(1)(1), "Not full LP A2")

    try {
      Match.solveTransportationLp(combined, Array(0, 0), forceFull = true)
      assertTrue(false, "Should have thrown LP exception")
    } catch {
      case e: RuntimeException => // Expected
    }

    // 3. relocationNote
    val users = List(
      User(1, "A", 0.0, 0.0, List("python")),
      User(2, "B", 0.0, 0.0, List("c"))
    )
    val place = Place(1, "P", 0.0, 0.0, List("python"), Some(1))
    val note1 = Match.relocationNote(users, 0, place, Array(50.0, 10.0), Array(1.0, 0.4), 50.0, 1.0)
    assertTrue(note1.contains("exceeds the 30"), "Relocation note should have exceeds")
    assertTrue(note1.contains("beats the best in-range candidate, B"), "Should mention B")

    val note2 = Match.relocationNote(users, 0, place, Array(50.0, 60.0), Array(1.0, 0.4), 50.0, 1.0)
    assertTrue(note2.contains("No candidate lives within"), "Should mention no candidate")

    // 4. Test llmScore and API key
    Match.getLlmApiKey() // should not crash
    val user2 = User(10, "X", 0, 0, List("x"))
    val place2 = Place(20, "Y", 0, 0, List("y"), None)
    Match.llmScore("dummy", user2, place2)

    // 5. Test main without LLM
    println("Testing Match.main()...")
    Match.main(Array("--allow-unassigned"))
    Match.main(Array("--llm", "--allow-unassigned"))
    
    println("All Match.scala tests passed!")
  }
}

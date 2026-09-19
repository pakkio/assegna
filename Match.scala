import upickle.default._
import java.nio.file.{Files, Paths}
import scala.io.Source
import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.util.UUID
import org.apache.commons.math3.optim.linear._
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.MaxIter
import scala.jdk.CollectionConverters._

case class User(id: Int, name: String, lat: Double, lon: Double, capabilities: List[String])
object User {
  implicit val rw: ReadWriter[User] = readwriter[ujson.Value].bimap[User](
    u => ujson.Obj("id" -> ujson.Num(u.id.toDouble), "name" -> ujson.Str(u.name), "lat" -> ujson.Num(u.lat), "lon" -> ujson.Num(u.lon), "capabilities" -> ujson.Arr.from(u.capabilities.map(ujson.Str))),
    json => User(
      json("id").num.toInt,
      json("name").str,
      json("lat").num,
      json("lon").num,
      json("capabilities").arr.map(_.str).toList
    )
  )
}

case class Place(id: Int, name: String, lat: Double, lon: Double, required_capabilities: List[String], capacity: Option[Int] = None)
object Place {
  implicit val rw: ReadWriter[Place] = readwriter[ujson.Value].bimap[Place](
    p => ujson.Obj("id" -> ujson.Num(p.id.toDouble), "name" -> ujson.Str(p.name), "lat" -> ujson.Num(p.lat), "lon" -> ujson.Num(p.lon), "required_capabilities" -> ujson.Arr.from(p.required_capabilities.map(ujson.Str)), "capacity" -> ujson.Num(p.capacity.getOrElse(1).toDouble)),
    json => Place(
      json("id").num.toInt,
      json("name").str,
      json("lat").num,
      json("lon").num,
      json("required_capabilities").arr.map(_.str).toList,
      json.obj.get("capacity").map(_.num.toInt)
    )
  )
}


case class MatchResult(place: String, user: String, capabilityScore: Double, distanceKm: Double, note: Option[String] = None)

object Match {
  val MAX_RELOCATION_KM = 30.0
  val OVER_CAP_PENALTY_PER_KM = 0.005
  val EARTH_RADIUS_KM = 6371.0
  val LLM_THRESHOLD = 0.5
  val BASE_URL = "https://opencode.ai/zen/go/v1"

  def haversineKm(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double = {
    val p1 = Math.toRadians(lat1)
    val p2 = Math.toRadians(lat2)
    val dphi = Math.toRadians(lat2 - lat1)
    val dlambda = Math.toRadians(lon2 - lon1)
    val a = Math.pow(Math.sin(dphi / 2), 2) + Math.cos(p1) * Math.cos(p2) * Math.pow(Math.sin(dlambda / 2), 2)
    2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(a))
  }

  def loadJson[T: ReadWriter](path: String): List[T] = {
    val jsonString = Source.fromFile(path)("UTF-8").mkString
    read[List[T]](jsonString)
  }

  def getLlmApiKey(): Option[String] = {
    val envPath = Paths.get("..", ".env").toAbsolutePath.normalize()
    sys.env.get("OPENCODEGO_API_KEY").orElse {
      if (Files.exists(envPath)) {
        Source.fromFile(envPath.toFile).getLines()
          .find(_.startsWith("OPENCODEGO_API_KEY="))
          .map(_.substring("OPENCODEGO_API_KEY=".length).stripPrefix("\"").stripSuffix("\""))
      } else None
    }
  }

  def llmScore(apiKey: String, user: User, place: Place): Double = {
    val model = sys.env.getOrElse("OPENCODEGO_MODEL", "deepseek-v4-flash")
    val prompt = s"User capabilities: ${user.capabilities.mkString(", ")}\n" +
      s"Place '${place.name}' required capabilities: ${place.required_capabilities.mkString(", ")}\n" +
      "Rate how well this user semantically fits the place's requirements, from 0.0 (no fit) " +
      "to 1.0 (perfect fit), giving partial credit for related or adjacent skills even without " +
      "an exact string match. Reply with ONLY a number between 0.0 and 1.0."

    val body = ujson.Obj(
      "model" -> model,
      "messages" -> ujson.Arr(
        ujson.Obj("role" -> "user", "content" -> prompt)
      ),
      "temperature" -> 0
    )

    val sessionId = UUID.randomUUID().toString
    val req = HttpRequest.newBuilder()
      .uri(URI.create(s"$BASE_URL/chat/completions"))
      .header("Content-Type", "application/json")
      .header("Authorization", s"Bearer $apiKey")
      .header("x-opencode-session", sessionId)
      .POST(HttpRequest.BodyPublishers.ofString(body.render()))
      .build()

    val client = HttpClient.newHttpClient()
    val res = client.send(req, HttpResponse.BodyHandlers.ofString())
    
    if (res.statusCode() == 200) {
      try {
        val json = ujson.read(res.body())
        val text = json("choices")(0)("message")("content").str.trim
        val score = text.split("\\s+")(0).toDouble
        Math.max(0.0, Math.min(1.0, score))
      } catch {
        case _: Exception => 0.0
      }
    } else 0.0
  }

  def solveTransportationLp(combined: Array[Array[Double]], capacities: Array[Int], forceFull: Boolean): Array[Array[Boolean]] = {
    val n = combined.length
    val m = combined(0).length
    val numVars = n * m

    val objCoeffs = new Array[Double](numVars)
    for (i <- 0 until n; j <- 0 until m) objCoeffs(i * m + j) = -combined(i)(j)
    val objective = new LinearObjectiveFunction(objCoeffs, 0.0)

    val constraints = new java.util.ArrayList[LinearConstraint]()

    for (j <- 0 until m) {
      val coeffs = new Array[Double](numVars)
      for (i <- 0 until n) coeffs(i * m + j) = 1.0
      constraints.add(new LinearConstraint(coeffs, Relationship.LEQ, capacities(j).toDouble))
    }

    val userRel = if (forceFull) Relationship.EQ else Relationship.LEQ
    for (i <- 0 until n) {
      val coeffs = new Array[Double](numVars)
      for (j <- 0 until m) coeffs(i * m + j) = 1.0
      constraints.add(new LinearConstraint(coeffs, userRel, 1.0))
    }

    val solver = new SimplexSolver()
    try {
      val solution = solver.optimize(
        new MaxIter(100000),
        objective,
        new LinearConstraintSet(constraints),
        GoalType.MINIMIZE,
        new NonNegativeConstraint(true)
      )

      val assignment = Array.ofDim[Boolean](n, m)
      for (i <- 0 until n; j <- 0 until m) {
        if (solution.getPoint()(i * m + j) > 0.5) assignment(i)(j) = true
      }
      assignment
    } catch {
      case e: Exception => throw new RuntimeException(s"LP solver failed: ${e.getMessage}")
    }
  }

  def relocationNote(users: List[User], userIdx: Int, place: Place, distancesCol: Array[Double], capScoresCol: Array[Double], distance: Double, capScore: Double): String = {
    val candidates = users.indices.filter(i => i != userIdx && distancesCol(i) <= MAX_RELOCATION_KM)
    val userName = users(userIdx).name
    if (candidates.isEmpty) {
      f"No candidate lives within ${MAX_RELOCATION_KM}%.0fkm of ${place.name}; ${userName} is the closest available fit at ${distance}%.1fkm, so the cap was exceeded."
    } else {
      val bestI = candidates.maxBy(i => capScoresCol(i))
      val bestName = users(bestI).name
      val bestScore = capScoresCol(bestI)
      f"${userName} relocates ${distance}%.1fkm (exceeds the ${MAX_RELOCATION_KM}%.0fkm cap) because their capability fit (${capScore}%.2f) clearly beats the best in-range candidate, ${bestName} (fit ${bestScore}%.2f) within ${MAX_RELOCATION_KM}%.0fkm."
    }
  }

  def main(args: Array[String]): Unit = {
    val llm = args.contains("--llm")
    val allowUnassigned = args.contains("--allow-unassigned")
    
    val users = loadJson[User]("users.json")
    val places = loadJson[Place]("places.json")

    val apiKey = if (llm) getLlmApiKey().getOrElse(sys.error("OPENCODEGO_API_KEY not found")) else ""

    val distances = Array.ofDim[Double](users.size, places.size)
    val capScores = Array.ofDim[Double](users.size, places.size)

    for (i <- users.indices; j <- places.indices) {
      val u = users(i)
      val p = places(j)
      distances(i)(j) = haversineKm(u.lat, u.lon, p.lat, p.lon)
      
      val uCaps = u.capabilities.map(_.toLowerCase).toSet
      val pCaps = p.required_capabilities.map(_.toLowerCase).toSet
      var base = if (pCaps.isEmpty) 0.0 else uCaps.intersect(pCaps).size.toDouble / pCaps.size
      
      if (llm && base < LLM_THRESHOLD) {
        base = Math.max(base, llmScore(apiKey, u, p))
      }
      capScores(i)(j) = base
    }

    val capacities = places.map(_.capacity.getOrElse(1)).toArray
    val totalCapacity = capacities.sum
    var wantFull = !allowUnassigned

    if (wantFull && totalCapacity < users.size) {
      println(s"(total capacity $totalCapacity < ${users.size} users -- can't force full assignment, falling back to best-available)\n")
      wantFull = false
    }

    val combined = Array.ofDim[Double](users.size, places.size)
    for (i <- users.indices; j <- places.indices) {
      val overCap = Math.max(0.0, distances(i)(j) - MAX_RELOCATION_KM)
      combined(i)(j) = capScores(i)(j) - overCap * OVER_CAP_PENALTY_PER_KM
    }

    val assignment = solveTransportationLp(combined, capacities, wantFull)
    var results = List[MatchResult]()

    for (i <- users.indices; j <- places.indices) {
      if (assignment(i)(j)) {
        val dist = distances(i)(j)
        val score = capScores(i)(j)
        val note = if (dist > MAX_RELOCATION_KM) {
          Some(relocationNote(users, i, places(j), distances.map(_(j)), capScores.map(_(j)), dist, score))
        } else None
        
        results ::= MatchResult(
          places(j).name,
          users(i).name,
          BigDecimal(score).setScale(3, BigDecimal.RoundingMode.HALF_UP).toDouble,
          BigDecimal(dist).setScale(1, BigDecimal.RoundingMode.HALF_UP).toDouble,
          note
        )
      }
    }

    val sortedResults = results.sortBy(r => (r.place, -r.capabilityScore))
    val unmatched = users.size - sortedResults.size
    
    if (unmatched > 0) {
      println(s"($unmatched user(s) left unassigned -- no place had net-positive value for them)\n")
    }

    for (m <- sortedResults) {
      val placeStr = m.place.padTo(30, ' ')
      val userStr = m.user.padTo(20, ' ')
      println(s"$placeStr -> $userStr (fit=${m.capabilityScore}, dist=${m.distanceKm}km)")
      m.note.foreach(n => println(s"  note: $n"))
    }
  }
}

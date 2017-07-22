import SANSA.graph.FactorGraph

import ml.dmlc.mxnet.Symbol
object factorGraphTest {

	def main(args: Array[String]): Unit = {
//		val rule = "t_stress(P,Yes) :- assign_yes(Yes),person(P),any(P,Yes) {r1}."
		val rule = "cancer(P,Yes) :- smokes(P,Yes), cancer_smoke(P,Yes ) {r8}."
		var g = new FactorGraph(rule, 10)
//		println("the variables and factors are")
//		println(g.variables)
//		println(g.factors)
//
//		println("The variables are: ")
//		for (variable <- g.variables) {
//			println(variable.label)
//		}
//		println("The factors are: ")
//		for (factor <- g.factors) {
//			println(factor.label)
//		}

		val exec = g.beliefPropagation()
		println(exec.argDict)
		println(exec.gradDict)

//		val neighbors = g.getNeighbors(g.variables(2),g.factors(1))
//		println(neighbors.length)

//		val a = Symbol.Variable("poop")
//		val b = Symbol.Variable("tatti")
//		Symbol.dot("dot")()(Map("lhs" -> a, "rhs" -> b))
	}
}

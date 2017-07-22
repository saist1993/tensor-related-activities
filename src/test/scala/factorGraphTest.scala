import SANSA.graph.FactorGraph

import ml.dmlc.mxnet.Symbol
object factorGraphTest {

	def main(args: Array[String]): Unit = {

		val rule = "cancer(P,Yes) :- smokes(P,Yes), cancer_smoke(P,Yes ) {r8}."
		var g = new FactorGraph(rule, 10)

		val exec = g.beliefPropagation()
		println(exec.argDict)
		println(exec.gradDict)

	}
}

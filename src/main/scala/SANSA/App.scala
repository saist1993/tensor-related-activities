package SANSA

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}
import ml.dmlc.mxnet.{NDArray => nd}

import SANSA.graph.{Variable => v}
import SANSA.graph.{Factor => f}
import SANSA.FactorGraph

/**
  * @author ${user.name}
  */
object App {

	//  def foo(x : Array[String]) = x.foldLeft("")((a,b) => a + b)

	def main(args : Array[String]) {
	}
	//
	//  val x = s.Variable(name = "a")
	//  val w = s.Variable(name = "b")

	val vara = new v(false, false, s.Variable("a"), "a")
	//
	var vl = List(vara,new v(false, false, _u = s.Variable("b"), "b"),new v(false, false, _u = s.Variable("c"),"c"))
	var f1 = new f(_i = vara, _o = vl(1), _M = s.Variable("f1"), _label = "f1")
	var f2 = new f(_i = vara, _o = vl(2), _M = s.Variable("f2"), _label = "f2")
	var f3 = new f(_i = vl(2), _o = vl(1), _M = s.Variable("f3"), _label = "f3")

	var g = new FactorGraph(_variables = vl, _factors_body = List(f2,f3), _factor_head = f1)
	//  var n = g.getNeighbors(vl(2),exclude=f2)

	//  println(n(0)._label)
	//	var u = s.Variable("u")

	println("Jello")
	var op: Symbol = g.beliefPropagation()
	println(op.listArguments())

//	var a = List(1,2,3)
//	println(a(4))

}

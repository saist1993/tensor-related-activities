package SANSA

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}
import ml.dmlc.mxnet.{NDArray => nd}
import ml.dmlc.mxnet.module.Module

import scala.collection.mutable.MutableList

import SANSA.graph.{Variable => v}
import SANSA.graph.{Factor => f}
//import SANSA.FactorGraph

/**
  * @author ${user.name}
  */
object App {

	//  def foo(x : Array[String]) = x.foldLeft("")((a,b) => a + b)

	def main(args: Array[String]) {
	}

	//
	//  val x = s.Variable(name = "a")
	//  val w = s.Variable(name = "b")
	val numberOfEntities = 10

	val vara = new v(false, false, s.Variable("a", Map("type" -> "variable")), "a")
	//
	var vl = List(vara, new v(false, false, _u = s.Variable("b", Map("type" -> "variable")), "b"), new v(false, false, _u = s.Variable("c",  Map("type" -> "variable")), "c"))
	var f1 = new f(_i = vara, _o = vl(1), _M = s.Variable("f1", Map("type" -> "factor")), _label = "f1")
	var f2 = new f(_i = vara, _o = vl(2), _M = s.Variable("f2", Map("type" -> "factor")), _label = "f2")
	var f3 = new f(_i = vl(2), _o = vl(1), _M = s.Variable("f3", Map("type" -> "factor")), _label = "f3")

//	var g = new FactorGraph(_variables = vl, _factors_body = List(f2, f3), _factor_head = f1, _n_e = numberOfEntities)
//	//  var n = g.getNeighbors(vl(2),exclude=f2)
//
//	//  println(n(0)._label)
//	//	var u = s.Variable("u")
//
//	println("Jello")
//	var op = g.beliefPropagation()

//	println(op.listArguments())
//	println("op is:\t" + op.toString())
//
//
//	val r1 = nd.ones(10, 10)
//	val r2 = nd.ones(10, 10)
//	val e1 = nd.array(Array(0,0,1,0,0,1,1,0,0,1), shape = Shape(1,10))
//	val dr1 = nd.empty(10, 10)
//	val dr2 = nd.empty(10, 10)
//	val de1 = nd.empty(1, 10)


	////////////////////////////////////////////////BIND!///////////////////
	//Encapuslate op in a function
	//do a fwd and bkwd pass (or atleast compute gradients and loss expressions)
	//	var model = new Module(op)

	//	val exe = op.bind(ctx = Context.cpu(), args = Map("a" -> e1, "f2" -> r1, "f3" -> r2), argsGrad = Map("f1" -> dr1, "f2" -> dr2))
	//	exe.forward()
	//	println(exe.outputs)
	//	println(exe.outputs(0).shape)
	//	println(exe.outputs.length)
	////	model.bind(dataShapes = IndexedSeq(1,100),labelShapes = 1,)
	//
	////	var a = List(1,2,3)
	////	println(a(4))
	//	var gr1 = nd.ones(1,100)
	//	var gr2 = nd.ones(1,100)
	//	exe.backward( outGrads = Array(gr1, gr2 ))

	// Simple Bind
//	val exe = op.simpleBind(ctx = Context.cpu(), gradReq = "write", shapeDict = Map("a" -> Shape(1, 10), "f2" -> Shape(10, 10), "f3" -> Shape(10, 10)))
//	println(exe.argDict)
//	println(exe.gradDict)
//
//
//	exe.forward(true, ("a",e1), ("f2",r1), ("f3",r2))
//	println(exe.outputs(0).shape)
//	println(exe.outputs(0).toArray.mkString(" "))
//	exe.backward(outGrads = Array(de1, dr1, dr2))
//	println(dr1.toArray.mkString(" "))
//	println(r1.toArray.mkString(" "))
//	println(r1.toArray(0) - dr1.toArray(0))
//
////	exe.
//
//	println(op.getClass())
//	println(op.argDict)
//	println(op.gradDict)

	val rule = "t_stress(P,Yes) :- assign(Yes,yes),person(P) {r1}."

	def parseRules(_rule: String) = {
		/*
			Convert a rule into a list of string of variables and factors
			Output data:
				- List(String)
				- List(Tuple(String, String, String))
		 */

		var variables = MutableList[String]()
		var factors = MutableList[String]()
		var tokens = rule.split(":-")
		var head = tokens(0)
			//head = t_stress(P,Yes)
		var body = tokens(1)
			//body =  assign(Yes,yes),person(P) {r1}.

		//Fix the head
		var headFactor = head.split("\\(")(0)  //DONE!
			//headFactor = t_stress (DONE)
		var headRest = head.slice(headFactor.length,head.length)
			//headRest = (P,Yes)
		headRest = headRest.replace("(","").replace(")","")
			//headRest = P,Yes
		var headVars = headRest.split(",")    //DONE!
			//headVars.mkString(" and ") = P and Yes

		//Fix the body
		body = body.slice(0,body.length - 6).trim
			//body = assign(Yes,yes),person(P)
		var bodyTokens = body.split("\\),")
			//bodyTokens.mkString("|") = assign(Yes,yes|person(P)
		var bodyFactors = MutableList[String]()
		var bodyVariables = MutableList[String]()
		for (token <- bodyTokens) {

			//Get the tokenhead
			var tokenHead = token.split("\\(")(0).trim
				//tokenHead = person

			//Get the body
			var tokenBody = token.split("\\(")(1).trim
				//tokenBody = Yes,yes || P)
			if (tokenBody(tokenBody.length-1) == ')') {
				//If here, remove the trailing bracket
				tokenBody = tokenBody.slice(0,tokenBody.length-1)
			}
				//tokenBody = Yes,yes || P

			//Binary token
			if (tokenBody.contains(',')) {

				//Find the left and right variable.
				//Append both of them to a mutable map.

				var leftVar = tokenBody.split(',')(0).trim
				var rightVar = tokenBody.split(',')(1).trim

//				if !()
				bodyFactors += tokenHead

			}

//			println(tokenBody(tokenBody.length-1))
		}
	}


	println(parseRules(rule))
}



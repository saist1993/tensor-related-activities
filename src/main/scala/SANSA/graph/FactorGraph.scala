package SANSA.graph

import SANSA.graph.{Factor => f, Variable => v}
import ml.dmlc.mxnet.{NDArray => nd, Symbol => s, _}

import scala.collection.mutable.{ListBuffer, MutableList, Map => mutableMap}

/**
  * Created by priyansh on 10/07/17.

  * This class takes a bunch of variables and factors and creates a coherent graph out of them.
  * The graph is represented as an adjacency matrix.
  * Thereafter it runs the belief propagation on it,
  * as described in the algorithm in the Tensorlog paper (https://arxiv.org/pdf/1605.06523.pdf)
  *
  **/
class FactorGraph(_rule: String, _n_e: Int) {

	/*
		First conver the params into a list of Variable objects and Factor objects
	 */

	// Create an empty variable to be used for elegancy.
	private val blankNode = new v(_is_head = false, _is_blank = true, _u = null, _label = "phi")
	private val numberEntities = _n_e

	private val (variablesStr, factorsStr, headFactorStr) = parseRules(_rule)

	private var (variables, factors, headFactor) = convertToNodes(variablesStr, factorsStr, headFactorStr)

	def getNeighbors(node: v, exclude: f): Seq[f] = {
		/*
		  Internal Function.
				Returns the neighbouring factors of the given node.
				Can exclude a specific factor from it, if required.

				Tested. Okay :]
		*/

		var neighbors = new ListBuffer[f]()
		for (factor <- factors) {

		  if ( node == factor.i || node == factor.o ) {   // If node appears in the left or right of any factor
		    if ( factor != exclude ){                    // And if the selected factor is not the one we intend to exclude
		      neighbors += factor
		    }
		  }
		}

		// Convert the Listbuffer into a neat little list.
		neighbors
	}

	private def getSymbol(_label: String): s = {
		/*
			Use this function to get any symbol referred with a string.
			Usecase: finding symbols fitting the listArguments() output on our desired symbolic expression.
		 */
		for (variable <- variables) {
			if (variable.label.equals(_label)) {
				return variable.u
			}
		}

		for (factor <- factors) {
			if (factor.label.equals(_label)) {
				return factor.M
			}
		}

		return null
	}

	private def parseRules(_rule: String) = {
		/*
			Convert a rule into a list of string of variables and factors
			Output data:
				- List(String)
				- List(Tuple(String, String, String))
		 */

		var variables = MutableList[String]()
		var factors = mutableMap[String, (String, String)]()

		var tokens = _rule.split(":-")
		var head = tokens(0)
		//head = t_stress(P,Yes)
		var body = tokens(1)
		//body =  assign(Yes,yes),person(P) {r1}.

		//Fix the head
		var headfactor = head.split("\\(")(0)  //DONE!
		//headFactor = t_stress (DONE)
		var headRest = head.slice(headfactor.length,head.length)
		//headRest = (P,Yes)
		headRest = headRest.replace("(","").replace(")","")
		//headRest = P,Yes
		var headVars = headRest.split(",")    //DONE!
		for (headVar <- headVars) {
			if (!variables.contains(headVar)) variables += headVar.trim
		}

		var headFactor = if (headVars.length == 1) {
			Map(headfactor -> (headVars(0).trim, null))
		} else {
			Map[String, (String, String)](headfactor -> (headVars(0).trim, headVars(1).trim))
		}

		//headVars.mkString(" and ") = P and Yes

		//Fix the body
		body = body.slice(0,body.length - 6).trim
		//body = assign(Yes,yes),person(P)
		var bodyTokens = body.split("\\),")
		//bodyTokens.mkString("|") = assign(Yes,yes|person(P)

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


			if (tokenBody.contains(',')) {
				//Binary token

				//Find the left and right variable.
				//Append both of them to a mutable map.

				var leftVar = tokenBody.split(',')(0).trim
				var rightVar = tokenBody.split(',')(1).trim

				if (!variables.contains(leftVar)) variables += leftVar
				if (!variables.contains(rightVar)) variables += rightVar

				factors += (tokenHead -> (leftVar, rightVar))
			}
			else {
				//Unary token
				var leftVar = tokenBody.trim
				if (!variables.contains(leftVar)) variables += leftVar
				factors += (tokenHead -> (leftVar, null))
			}

		}

		(variables.toList, factors.toMap, headFactor)
	}

	private def convertToNodes(_variables: List[String], _factors: Map[String, (String, String)], _headFactorStr: Map[String, (String, String)]) =  {
		//This function is used to create nodes out of the given data.
		var variableMap = mutableMap[String, v]()

		var variableList = MutableList[v]()
		var factorList = MutableList[f]()

		for (variable <- _variables) {
			val temp = new v(false, false, s.Variable(variable, Map("type" -> "variable")), variable)
			variableMap += (variable -> temp )
			variableList += temp
		}

		for (factor <- _factors) {
			var leftVar = factor._2._1
			var rightVar = factor._2._2

			var leftv = variableMap(leftVar)
			if (rightVar != null) {
				//Boolean hai
				var rightv = variableMap(rightVar)

				factorList += new f(_i = leftv, _o = rightv, _M = s.Variable(factor._1, Map("type" -> "factor",  "unary" -> "no")), _label = factor._1)
			} else {
				var rightv = null
				factorList += new f(_i = leftv, _o = rightv, _M = s.Variable(factor._1, Map("type" -> "factor", "unary" -> "yes")), _label = factor._1)
			}

		}

		//head factor
		val hf = _headFactorStr.keys.toList(0)
		val (l,r) = _headFactorStr(_headFactorStr.keys.toList(0))

		val leftVar = variableMap(l)

		var headFactor = if (null != r) {
			//Binary hai
			var rightv = variableMap(r)
			new f(_i = leftVar, _o = rightv, _M = s.Variable(hf,Map("type" -> "factor",  "unary" -> "no")), _label = hf)
		} else {
			var rightv = null
			new f(_i = leftVar, _o = rightv, _M = s.Variable(hf,Map("type" -> "factor",  "unary" -> "no")), _label = hf)
		}

		(variableList, factorList, headFactor)
	}

	def beliefPropagation(): Executor = {
		/*
		  Call this function to receive expression achieved by running the belief propagation algorithm.
				We implement the algorithm listed in the paper mentioned in the comments above.

				Thereafter, we bind and compile the function and then return them back to whoever calls the function.

		  @TODO: We pass strings as of now, replace with symbolic ops ASAP!

				Pseudocode:
					-> Start with the o variable in head_factor,
					-> Call compile_message_node on it.
					-> Let the recursive functions do their thing and finally catch the composed expression
					-> Do a softmax over this expression
					-> Implement a cross entropy loss
						-> requires a new defined symbolic variable
					-> Implement square cost (or any other) over it.
					-> Bind this cost expression (try defining what to pull the gradients over, and what to ignore)
					-> Throw the executor back.
		 */
		//Receive the equation from Belief Propagation.
		var y = compileMessage_variable(headFactor.o, headFactor)

		//Declare a new symbol, to represent the true output
		var y_cap = s.Variable("true_output", Map("type" -> "variable"))

		//Using softmax(y) and y_cap, compute cross entropy loss
//		var loss = s.softmax_cross_entropy(name = "loss")()(Map("data" -> y, "label" -> y_cap))

		//Compute the cost
		var cost = s.LinearRegressionOutput(name = "cost")()(Map("data" -> y, "label" -> y_cap))

		//To create the bind, we need a bit of shape inferences.
			//First, programatically create a map of every factor name and its shape
		var map = mutableMap[String, Shape]()
		for (label <- cost.listArguments()) {
			var symbol = getSymbol(label)
			if (symbol != null) {
				//If the symbol is a variable
				if (symbol.attr("type") == Some("variable")) {
					map += (label -> Shape(1, numberEntities))
				}
				else if (symbol.attr("type") == Some("factor")) {
					if (symbol.attr("unary") == Some("yes")) {
						map += (label -> Shape(1, numberEntities))
					} else {
						map += (label -> Shape(numberEntities, numberEntities))
					}
				}
			}

		}
		map += ("true_output" -> Shape(1,numberEntities))

			//Then use that map to bind the cost symbol.
		//@TODO: See which keyword best suites our purposes on gradReq
		var executor = cost.simpleBind(ctx = Context.cpu(), gradReq = "write", shapeDict = map.toMap)

		executor
	}

	private def compileMessage_variable(_node: v, _factor: f): s = {
		/*
		  Pseudocode:
		  (treat _node as X and _factor as L )

		  if X is the input variable (global) then
		    return u_c , the input

		  else
		    generate a new variable name v_x
		    collect neighbouring L_i of X excluding L
		    for [L_1, L_2 .. L_i ], do
		      v_i = compile_message(L_i -> X)
		    emit(v_x = v1 dot v2 ... dot vi)

		    return v_x
		 */

		if ( _node == headFactor.i )  return _node.u


		// Get the neighbors of this node, except for the factor given in the args
		val neighbors = getNeighbors(_node, exclude = _factor)

		// Send the neighbour + current node to compilemessage_factor and collect what they have to say.
		var neighbor_values = new ListBuffer[s]()
		for (factor <- neighbors) {
			var temp: s = compileMessage_factor(factor, _node)
			neighbor_values.+=(temp)
		}

		/*
			This code block does elementwise multiplication of all the values received in neighbor_values
		*/
		var v_x: s = null
		try {
			v_x = neighbor_values(0)
		} catch {

			//This implies that there are no neighbors. We should not come across something like this. Pray to god nothing's broken
			case e: Exception => {
				//@TODO: Handle this case.
				v_x = null
			}
		}
		// For the rest of the symbols (if something still remains), do an elementwise multiplication to v_x
		for (symbol <- neighbor_values.slice(1,neighbor_values.length)) {
			v_x = v_x * symbol
		}

		//Return it.
		v_x
	}

	private def compileMessage_factor(_factor: f, _node: v): s = {
		/*
		Pseudocode:
			(treat _node as X and _factor as L)

			if L is a unary factor:
				emit v_L,_X = v_L

			elif X is the output node of L
				v_i = compilemessage_node(X_o, L)

			elif X is the output node of L
				v_i =  compilemessage_node(X_i, L)

			return this
		*/
		var v_i: s = null

		//@TODO: See if we need another variable like self.v to represent the value of unary predicates.
		if (_factor.o == null)  return _factor.M                                            // If the factor is unary.

		//The factor then must be a good ol' binary factor
		else if (_factor.o == _node) v_i = compileMessage_variable(_factor.i, _factor)      // If the node is the output node for this factor

		else v_i = compileMessage_variable(_factor.o, _factor)                              // If the node is the input node for this factor

		return s.dot("dot")()(Map("lhs" -> v_i, "rhs" -> _factor.M))
	}
}

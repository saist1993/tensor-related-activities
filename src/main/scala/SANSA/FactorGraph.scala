package SANSA

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}
import ml.dmlc.mxnet.{NDArray => nd}

import scala.collection.mutable.ListBuffer

import SANSA.graph.{Factor => f, Variable => v}

/**
  * Created by priyansh on 10/07/17.

  * This class takes a bunch of variables and factors and creates a coherent graph out of them.
  * The graph is represented as an adjacency matrix.
  * Thereafter it runs the belief propagation on it,
  * as described in the algorithm in the Tensorlog paper (https://arxiv.org/pdf/1605.06523.pdf)
  *
  **/
class FactorGraph(_variables: List[v], _factors_body: List[f], _factor_head: f) {

	/*
	Copy the parameters into internal variables.
	*/

	// Create an empty variable to be used for elegancy.
	private val blank_node = new v(_is_head = false, _is_blank = true, _label = "phi")

	// Keep the list of variables in order to map the indices of the matrix to actual graph nodes
	private val variables = _variables :+ blank_node
	private val factors = _factors_body

	// Keep the predicate that appeared in the head of the query in order to know where do we start the BP
	private val head_factor = _factor_head

	private def getNeighbors(node: v, exclude: f): Seq[f] = {
		/*
		  Internal Function.
				Returns the neighbouring factors of the given node.
				Can exclude a specific factor from it, if required.

				Tested. Okay :]
		*/
		var neighbors = new ListBuffer[f]()

		for (factor <- factors) {

		  if ( node == factor.i || node == factor.o ){   // If node appears in the left or right of any factor
		    if ( factor != exclude ){                    // And if the selected factor is not the one we intend to exclude

		      neighbors += factor

		    }
		  }
		}

		// Convert the Listbuffer into a neat little list.
		neighbors
	}

	def beliefPropagation(): String = {
		/*
		  Call this function to receive expression achieved by running the belief propagation algorithm.
				We implement the algorithm listed in the paper mentioned in the comments above.

				Thereafter, we bind and compile the function and then return them back to whoever calls the function.

		  @TODO: We pass strings as of now, replace with symbolic ops ASAP!

				Pseudocode:
					-> Start with the o variable in head_factor,
					-> Call compile_message_node on it.
					-> Let the recursive functions do their thing and finally catch the composed expression
					-> Bind the expression to a function and throw it back.
		 */

		val output = compileMessage_variable(head_factor.o, head_factor)
		output
	}

	private def compileMessage_variable(_node: v, _factor: f): String = {
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

		if ( _node == head_factor.i ) return "u_c"


		// Get the neighbors of this node, except for the factor given in the args
		val neighbors = getNeighbors(_node, exclude = _factor)

		// Send the neighbour + current node to compilemessage_factor and collect what they have to say.
		val neighbor_values = new ListBuffer[String]()
		for (factor <- neighbors) {
		  neighbor_values += compileMessage_factor(factor, _node)
		}

		// Do a scala equivalent of " /dot ".join(neighbor_values) -_-
		var v_x: String = ""
		for (i <- 0 until neighbor_values.length) {
			v_x = v_x.concat(neighbor_values(i))
			if ( i != neighbor_values.length-1 ) {
				v_x = v_x.concat(" /mul ")
			}
		}


		// val v_x: String = neighbor_values.flatten mkString " /mul ". Nopes. Does not work.

		v_x
	}

  private def compileMessage_factor(_factor: f, _node: v): String = {
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

    // If the factor is unary. @TODO: See if we need another variable like self.v to represent the value of unary predicates.
    if (_factor.o == None)  return _factor.label

    else if (_factor.o == _node) {
		// If the node is the output node for this factor
		val v_i: String = compileMessage_variable(_factor.i, _factor)
		return v_i + " /dot " + _factor.label
    }

	else if (_factor.i == _node) {
		// If the node is the input node for this factor
		val v_i: String = compileMessage_variable(_factor.o, _factor)
		return v_i + " /dot " + _factor.label
	}

	else {
	    // Code should not come to this block. Something wrong with the algorithm.
	    // This one is here to make the Scala compiler happy.
	    // @TODO: Shoot a warning, will ye
	    return "Poop"
    }



  }
}

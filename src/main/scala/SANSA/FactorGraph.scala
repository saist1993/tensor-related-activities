package SANSA

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}
import ml.dmlc.mxnet.{NDArray => nd}

import scala.collection.mutable.ListBuffer

import SANSA.graph.{Factor => f, Variable => v}

/**
  * Created by priyansh on 10/07/17.
  *
  * This class takes a bunch of variables and factors and creates a coherent graph out of them.
  * The graph is represented as an adjacency matrix.
  * Thereafter it runs the belief propagation on it,
  * as described in the algorithm in the Tensorlog paper (https://arxiv.org/pdf/1605.06523.pdf)
  *
  */
class FactorGraph(_variables: Array[v], _factors: Array[f], _fictional_factor: f) {

  /*
    Copy the parameters into internal variables.
   */

  // Create an empty variable to be used for elegsancy.
  private val blank_node = new v(_is_head = false, _is_blank = true, _label = "phi")

  // Keep the list of variables in order to map the indices of the matrix to actual graph nodes
  private val variables = _variables :+ blank_node
  private val factors = _factors

  // Keep the predicate that appeared in the head of the query in order to know where do we start the BP
  private val head_predicate = _fictional_factor

//  def buildGraph(): Unit = {
//    /*
//        This method generates a graph out of given data, in the form of an adjacency matrix.
//        Pseudocode:
//          -> create a matrix of zeros, of n_v * n_v dimensions.
//          -> loop over all the factors and based on the position of the i and o of the factor, place the factor object there.
//     */
//
//    var graph = nd.zeros(variables.length,variables.length)
//    for(factor <- factors){
//      graph(variables.indexOf(factor.i))(variables.indexOf(factor.o)) = factor
//    }
//
//  }


  private def getNeighbors(node: v, exclude: f): Seq[f] = {
    /*
      Internal Function.
			Returns the neighbouring factors of the given node.
			Can exclude a specific factor from it, if required.
    */

    var neighbors = new ListBuffer[f]()

    for(factor <- factors) {
      if( node == factor.i || node == factor.o ){   //If node appears in the left or right of any factor
        if( factor != exclude ){                    //And if the selected factor is not the one we intend to exclude
          neighbors += factor
        }
      }
    }

    //Convert the factor into
  }
}

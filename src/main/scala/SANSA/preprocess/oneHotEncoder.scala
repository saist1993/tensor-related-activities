package SANSA.preprocess

import ml.dmlc.mxnet.{Shape, NDArray => nd}

import scala.collection.mutable.MutableList


//@TODO: make it serializable before throwing it to spark map:
object oneHotEncoder  extends Serializable {
	/*
		This object is used to one-hot encode everything that is passed to it.
		Give it an int (typically, representing an entity) and it will give you a one-hot representation back.

		Sorry for the bad code structure, but this will just consist of a function which does the job.

		@TODO: Instead of one-hot, you need floats here (probabilities, in case you're dealing with probabilistic databases.
		Only parameter: dimensions.
	 */

	def encode(_input: (List[Int], Int)): (Int, List[Array[Float]]) = {
		var triple = _input._1
		var dims = _input._2

//	def encode(_index: Int, _dimensions: Int): nd = {
		/*
			Pseudocode: make an array using the scala fill method. (all zeros)
			Access the desired position via the index and turn it to one.
		 */

		val arr = Array.fill(dims){0.0f}
		var oneHotted = MutableList[Array[Float]]()

		if (triple.length == 3) {
			//@TODO: nd.array(s, shape=Shape(1,dims)) -> s
			var s = arr.clone()
			s(triple(1)) = 1.0f
			oneHotted += s
			var o = arr.clone()
			o(triple(2)) = 1.0f
			oneHotted += o
		} else {
			var s = arr.clone()
			s(triple(1)) = 1.0f
			oneHotted += s
		}

		(triple(0), oneHotted.toList)

	}




}

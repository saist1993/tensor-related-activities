package SANSA.preprocess

import ml.dmlc.mxnet.{Shape, NDArray => nd}


//@TODO: make it serializable before throwing it to spark map
object oneHotEncoder {
	/*
		This object is used to one-hot encode everything that is passed to it.
		Give it an int (typically, representing an entity) and it will give you a one-hot representation back.

		Sorry for the bad code structure, but this will just consist of a function which does the job.

		@TODO: Instead of one-hot, you need floats here (probabilities, in case you're dealing with probabilistic databases.
		Only parameter: dimensions.
	 */

	def encode(_index: Int, _dimensions: Int): nd = {
		/*
			Pseudocode: make an array using the scala fill method. (all zeros)
			Access the desired position via the index and turn it to one.
		 */

		var output = Array.fill(_dimensions){0.0f}
		output(_index) = 1.0f
		nd.array(output, shape=Shape(1,_dimensions))
	}




}

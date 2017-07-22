package SANSA.preprocess

import ml.dmlc.mxnet.{NDArray => nd}
import scala.collection.mutable.{Map => mutableMap}

object dataCollector extends Serializable {
	/*
		This class is responsible for collecting data and create sparse matrices or vectors
			corresponding to the triples.

		One object of this class ought to have been made for one predicate and thus would store a
		matrix corresponding to that predicate.
	 */

	var data = mutableMap[Int, relationMatrix]()
	var numberOfEntities = 0


	def setNumberOfEntities(_val: Int): Unit = {
		if (_val > numberOfEntities) numberOfEntities = _val
	}

	def collect(_triple : List[Int]): List[Int] = {
		/*
			This function assumes a static object within which it resides.
			It takes one triple at a time and augments its corresponding sparse matrix depending upon the s, p, o
		 */

		//First check if unary or binary.
		val unary = if (_triple.length == 2) true else false

		//Then check if you already have the predicate and its matrix
		var matrix = {
			if (data.keys.toList.contains(_triple(0))) 	data(_triple(0))
			else {
				if (unary) {
					//@TODO: Check if you want it to be a (1,n) or a (n,1)
					data += (_triple(0) -> new relationMatrix(numberOfEntities, true))
					data(_triple(0))
//					nd.zeros(numberOfEntities,1)
				}
				else {
					data += (_triple(0) -> new relationMatrix(numberOfEntities, false))
					data(_triple(0))
//					nd.zeros(numberOfEntities, numberOfEntities)
				}
			}
		}

		if (unary) matrix.setValueUnary( 1.0f,_triple(0))
		else matrix.setValueBinary(1.0f, (_triple(1),_triple(2)))

		_triple

	}



}
package SANSA.preprocess

import ml.dmlc.mxnet.NDArray.array
import ml.dmlc.mxnet.{Shape, NDArray => nd}

class relationMatrix (val _dim: Int, val _unary: Boolean) {
	/*
		This class encapsulates an array (can be of one or two dimensions) based on what's required.
	 */

	var unary = _unary
	private var data_2d = Array.ofDim[Float](_dim,_dim)

	private var data_1d = Array.ofDim[Float](_dim)

	def setValueBinary(_value: Float, _index: (Int,Int)): Unit = {
		unary = false
		data_2d(_index._1)(_index._2) = _value
	}

	def setMatrixBinary(_matrix : Array[Array[Float]]): Unit ={
		unary = false
		data_2d = _matrix
	}

	def setValueUnary(_value: Float, _index: Int): Unit = {
		unary = true
		data_1d(_index) = _value
	}

	def setMatrixUnary(_matrix : Array[Float]): Unit ={
		unary = true
		data_1d = _matrix
	}

	def getMatrix() = {
		if (unary) data_1d
		else data_2d
	}

}

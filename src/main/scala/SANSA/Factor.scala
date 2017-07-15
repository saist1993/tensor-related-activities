package SANSA.graph

import ml.dmlc.mxnet.Symbol

/**
  * Created by priyansh on 10/07/17.
  */
class Factor( var _i: Variable, var _o: Variable, var _M:Symbol, override val _label: String)
	extends Node(_label) {

	var i: Variable = _i    //Variable on the left/in/from
	var o: Variable = _o    //Variable on the right/out/to
	var M: Symbol = _M //Replace this with a symbolic variablelater.


}

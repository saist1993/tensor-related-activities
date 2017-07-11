package SANSA.graph

/**
  * Created by priyansh on 10/07/17.
  */
class Factor( var _i: Variable, var _o: Variable, override val _label: String)
  extends Node(_label) {

  var i: Variable = _i    //Variable on the left/in/from
  var o: Variable = _o    //Variable on the right/out/to
  var M = null //Replace this with a symbolic variablelater.


}

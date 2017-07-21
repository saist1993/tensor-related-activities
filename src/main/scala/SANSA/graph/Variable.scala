package SANSA.graph

import ml.dmlc.mxnet.Symbol

/**
  * Created by priyansh on 10/07/17.
  */
class Variable ( val _is_head: Boolean, val _is_blank: Boolean, val _u: Symbol, override val _label: String)
    extends Node(_label) {

	val head: Boolean = _is_head
    val blank: Boolean = _is_blank
    var u: Symbol = _u

}

package SANSA.graph

/**
  * Created by priyansh on 10/07/17.
  */
class Variable ( val _is_head: Boolean, val _is_blank: Boolean, override val _label: String)
  extends Node(_label) {

  val head: Boolean = _is_head
  val blank: Boolean = _is_blank
  var u = null

}

package SANSA.preprocess

import scala.collection.mutable.{Map => mutableMap}

/**
  * Created by gaurav on 21.07.17.
  */
object labelEncoder extends Serializable {

	/*
		Expecting these four things here to remain static across map calls magically. Might throw a bug. Beware.
		@TODO: Optimize both the functions. I'm sure it can be made more elegant.
	 */
	var entityCounter = 0
	var predicateCounter = 0
	var entityEncoder = mutableMap[String, Int]()
	var predicateEncoder = mutableMap[String, Int]()

	/*
	Handle the entities differently.
	 */
	private def _encodeEntities(_input: String): Int = {
		var index: Int = -1
		try {
			index = entityEncoder(_input)
		} catch {
			case e: Exception => {
				entityEncoder += (_input -> entityCounter)
				entityCounter += 1
				index = entityEncoder(_input)
			}
		} finally {
			index
		}

		index
	}

	private def _encodePredicates(_input: String): Int = {
		var index: Int = -1
		try {
			index = predicateEncoder(_input)
		} catch {
			case e: Exception => {
				predicateEncoder += (_input -> predicateCounter)
				predicateCounter += 1
				index = predicateEncoder(_input)
			}
		} finally {
			index
		}

		index
	}

	def encode(_inputs: List[String]): List[Int] = {

		val encoded = List( _encodePredicates(_inputs(0)), _encodeEntities(_inputs(1)), _encodeEntities(_inputs(2)) )
		encoded
	}
}

package samples

import SANSA.preprocess.relationMatrix
import SANSA.preprocess.dataCollector

object dataCollectorTest {

	def main(args: Array[String]): Unit = {

		checkDataCollector()

	}

	private def checkRelationMatrix() = {
		//Checked. Works fine.

		//First lets test the datacollector.
		val mat = new relationMatrix( 5, false)
		mat.setValueBinary(2,(4,1))
		println(mat.getMatrix().deep.mkString("|"))
		println(mat.unary)
	}

	private def checkDataCollector() = {
		dataCollector.setNumberOfEntities(5)
		dataCollector.collect(List(2,1,4))
		dataCollector.collect(List(2,3,4))
		dataCollector.collect(List(2,0,0))
		dataCollector.collect(List(1,0,0))
		println(dataCollector.data(1).getMatrix().deep.mkString("|"))
		println(dataCollector.data(2).getMatrix().deep.mkString("|"))
//		println(dataCollector.data(2))
//		println(dataCollector.data(2))

	}

}

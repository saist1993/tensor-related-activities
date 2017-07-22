import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

import SANSA.preprocess.labelEncoder

object labelEncodingTest {

	def main(args: Array[String]): Unit = {

		val appName: String = """TensorLog"""
		val conf = new SparkConf().setAppName(appName).setMaster("local")
		val sc = new SparkContext(conf)

		val spark = SparkSession.builder().master("local").getOrCreate()
		import spark.implicits._
		val lines = sc.textFile("src/main/resources/smokers/raw/friends.cfacts")

		val data = lines.map(_.split("\t").to[List])

		val labelEncoding = data.map(labelEncoder.encode)



		labelEncoding.collect().foreach(println)
		println(labelEncoder.entityCounter)
	}

}

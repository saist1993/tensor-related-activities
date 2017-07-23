import SANSA.preprocess.labelEncoder
import SANSA.preprocess.dataCollector
import SANSA.preprocess.oneHotEncoder
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

object preProcessingTest {
	def main(args: Array[String]): Unit = {
		val appName: String = """TensorLog"""
		val conf = new SparkConf().setAppName(appName).setMaster("local")
		val sc = new SparkContext(conf)

		val spark = SparkSession.builder().master("local").getOrCreate()

		import spark.implicits._
		val rawRDD = sc.textFile("src/main/resources/smokers/smokers.cfacts")

		val parsedRDD = rawRDD.map(_.split("\t").to[List])
		var labelEncodedRDD = parsedRDD.map(labelEncoder.encode)

		labelEncodedRDD.collect().foreach(println)

		dataCollector.setNumberOfEntities(labelEncoder.entityCounter)
		labelEncodedRDD = labelEncodedRDD.map(dataCollector.collect)                    //Data collecter makes no changes to the RDD
		val oneHotEncodedRDD = labelEncodedRDD.map(s => (s,labelEncoder.entityCounter)).map(oneHotEncoder.encode)    //RDD is now.
		//@TODO: The one-hots are not nd array encapsulated as of yet. Beware!

		oneHotEncodedRDD.collect().foreach(println)
		println(labelEncoder.entityCounter)
	}
}

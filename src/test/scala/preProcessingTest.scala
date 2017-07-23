import SANSA.preprocess.labelEncoder
import SANSA.preprocess.dataCollector
import SANSA.preprocess.oneHotEncoder
import SANSA.graph.FactorGraph
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.optimizer.SGD

import scala.collection.mutable.{ListBuffer, MutableList, Map => MutableMap}


object preProcessingTest {
	def main(args: Array[String]): Unit = {

		/*
			Pre-processing timestep
		 */
		val appName: String = """TensorLog"""
		val conf = new SparkConf().setAppName(appName).setMaster("local")
		val sc = new SparkContext(conf)

		val spark = SparkSession.builder().master("local").getOrCreate()

		import spark.implicits._
		val rawRDD = sc.textFile("src/main/resources/smokers/smokers.cfacts")

		val parsedRDD = rawRDD.map(_.split("\t").to[List])
		var labelEncodedRDD = parsedRDD.map(labelEncoder.encode)

		labelEncodedRDD.collect()

		dataCollector.setNumberOfEntities(labelEncoder.entityCounter)
		labelEncodedRDD = labelEncodedRDD.map(dataCollector.collect)                    //Data collecter makes no changes to the RDD
		val oneHotEncodedRDD = labelEncodedRDD.map(s => (s,labelEncoder.entityCounter)).map(oneHotEncoder.encode)    //RDD is now.
		//@TODO: The one-hots are not nd array encapsulated as of yet. Beware!

		oneHotEncodedRDD.collect()

		/*
			Setting up the mathematical expression now.

			So far:
				Spark based preprocessing works well and now
				oneHotEncodedRDD, and some static objects hold all the required data.

			Now:
				make a new graph for every rule, get its belief prop expression (as mxnet executors)
				and create a (rule head -> executor) map for training.
		 */

		//Declare a map of headfactor's string and its corresponding list of executors
		var ruleBook = MutableMap[String, MutableList[Executor]]()

		//Open the rule file and split by new line.
		val rulesRDD = sc.textFile("src/main/resources/smokers/smokers.ppr")
		val rules = rulesRDD.collect()

		for (rule <- rules) {
			//Append the rules to the rulebook with the executor
			val g = new FactorGraph(rule, labelEncoder.entityCounter)
			val head = g.headFactorStr.keys.toList(0)

			if (ruleBook.keys.toList.contains(head)) ruleBook(head) += g.beliefPropagation()
			else {
				ruleBook += (head -> MutableList(g.beliefPropagation()))
			}

		}


		/*
			Training time!
		 */




	}
}

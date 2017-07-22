package SANSA

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}
import ml.dmlc.mxnet.{NDArray => nd}
import ml.dmlc.mxnet.module.Module
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{IndexToString, OneHotEncoder, StringIndexer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

import scala.collection.mutable.MutableList
//import SANSA.graph.{FactorGraph, Factor => f, Variable => v}
import SANSA.preprocess.{labelEncoder => lEnc}

/**
  * @author Priyansh Trivedi (geraltofrivia), Gaurav Maheshwari (saist1993)
  */
object App {

	def main(args: Array[String]): Unit = {
//		val appName: String = "Poop"
//		val conf = new SparkConf().setAppName(appName).setMaster("local")
//		val sc = new SparkContext(conf)
		val rule = "t_stress(P,Yes) :- assign(Yes,yes),person(P) {r1}."
//		var g = new FactorGraph(rule, 10)
//		println(g.variables)
//		println(g.factors)

	}

}



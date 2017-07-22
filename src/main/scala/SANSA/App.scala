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
		var mat = nd.zeros(10,10)
//		mat.set()
	}

}



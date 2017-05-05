package src.main.scala

import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Created by rajsarka on 5/5/2017.
  */
object test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    .setAppName("test")
    .setMaster("local[*]")

    val spark = SparkSession.builder()
    .config(conf)
    .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
    .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")

    training.show()

  }

}

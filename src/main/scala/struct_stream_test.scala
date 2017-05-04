package main.scala

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._


/**
  * Created by rajsarka on 5/4/2017.
  */
object struct_stream_test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("Structure streaming test")

    val spark = SparkSession
    .builder()
    .config(conf)
    .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
    .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    val filePath = "C:/Downloads/iot_dataset/all_data/"

    val csvSchema = new StructType(Array(
      StructField(name = "datetime", dataType = TimestampType),
      StructField(name = "user_id", dataType = StringType),
      StructField(name = "latitude", dataType = DoubleType),
      StructField(name = "longitude", dataType = DoubleType),
      StructField(name = "pulse", dataType = FloatType),
      StructField(name = "temp", dataType = FloatType),
      StructField(name = "age", dataType = IntegerType),
      StructField(name = "bp_category", dataType = StringType)
    ))

    val staticInput = spark.read
    .schema(csvSchema)
    .csv(filePath)

    staticInput.show(5)

    val categorized_bp_cat_indexer = new StringIndexer()
    .setInputCol("bp_category")
    .setOutputCol("bp_category_label")
    .fit(staticInput)

    val data_labelised = categorized_bp_cat_indexer.transform(staticInput)
    .select("latitude", "longitude", "pulse", "temp", "age", "bp_category_label")

    data_labelised.show(10)

    data_labelised.printSchema()

    val vectorised_data = data_labelised.rdd.map(row => {
      val label = row.getAs[Double]("bp_category_label")
      val latitude = row.getDouble(0)
      val longitude = row.getDouble(1)
      val pulse = row.getFloat(2)
      val temp = row.getFloat(3)
      val age = row.getInt(4)

      LabeledPoint(label, Vectors.dense(latitude, longitude, pulse, temp, age))
    })

    vectorised_data.take(5).foreach(println)



/*
    val staticCountDf = staticInput
    .groupBy($"bp_category", window(timeColumn = $"datetime", windowDuration = "5 minutes"))
    .count()

    staticCountDf.show(10)
*/


  }

}

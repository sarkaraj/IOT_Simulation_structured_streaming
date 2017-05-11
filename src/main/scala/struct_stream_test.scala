package src.main.scala

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import src.main.scala.dataModeler._
import src.main.scala.dataVectorise._


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

//    import spark.implicits._

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    val filePath = "C:/Downloads/iot_dataset/all_data/"

    val csvSchema = defineSchema()

    val staticInput = spark.read
    .schema(csvSchema)
    .csv(filePath)

    staticInput.show(5)

    val vector_input = vectorizeData(staticInput)

//    obtainLogisticModel(vector_input)

//    obtainRandomForestModel(vector_input)

    val results = loadRandomForestModel("models_/rfc_20").transform(vector_input.select("features"))

    results.printSchema()

    results.select("prediction").distinct().show()

  }



}

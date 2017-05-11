package src.main.scala

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import src.main.scala.classification.dataModelerClassification._
import src.main.scala.clustering.dataModelerClustering._
import src.main.scala.dataTransform.dataVectorise._


/**
  * Created by rajsarka on 5/4/2017.
  */
object modelTraining {
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

//    val sc = spark.sparkContext

    Logger.getRootLogger().setLevel(Level.ERROR)

    val filePath = "C:/Downloads/iot_dataset/all_data/"

    val csvSchema = defineSchema()

    val staticInput = spark.read
    .schema(csvSchema)
    .csv(filePath)

    staticInput.show(5)

    val vector_input = vectorizeUserBodyData(staticInput)
    val latlon_input = vectorizeUserLatLonData(staticInput)


    obtainLogisticModel(vector_input)
    obtainRandomForestModel(vector_input)
    obtainKmeansClusterModel(latlon_input)

/*

    val results = loadRandomForestModel("models_/rfc_20").transform(vector_input.select("features"))
    results.printSchema()
    results.select("prediction").distinct().show()

    val kmeansModel = loadKmeansClusterModel("models_/kmeans_6")

    kmeansModel.clusterCenters.foreach(println)

    val datafit = kmeansModel.transform(latlon_input)

    datafit.printSchema()

*/
//    datafit.select("prediction").show(50)





  }



}

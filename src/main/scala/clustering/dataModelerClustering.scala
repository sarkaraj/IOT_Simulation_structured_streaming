package src.main.scala.clustering

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by rajsarka on 5/11/2017.
  */
object dataModelerClustering {

  def obtainKmeansClusterModel(vectorisedData : DataFrame) : Unit = {
    val kmeans_data = new KMeans()
    .setK(6)
//    .setSeed(1L)
    .fit(vectorisedData)

    kmeans_data.write.overwrite().save("models_/kmeans_6")
  }

  def loadKmeansClusterModel(path : String) : KMeansModel = {
    val model = KMeansModel.load(path)
    model
  }

}

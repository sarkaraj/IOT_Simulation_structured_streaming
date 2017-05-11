package src.main.scala

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.DataFrame

/**
  * Created by rajsarka on 5/11/2017.
  */
object dataModeler {
  def obtainLogisticModel(vectorisedData : DataFrame) : Unit = {

    val mlr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.4)
      .setElasticNetParam(0.8)
//      .setFamily("multinomial")
      .fit(vectorisedData)

    mlr.write.overwrite().save("models_/mlr_10_0-4_0-8")
  }

  def loadLogisticModel(path : String) : LogisticRegressionModel = {
    val model = LogisticRegressionModel.load(path)
    model
  }

  def obtainRandomForestModel(vectorisedaData : DataFrame) : Unit = {

    val rfc = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(20)
    .fit(vectorisedaData)

    rfc.write.overwrite().save("models_/rfc_20")
  }

  def loadRandomForestModel(path : String) : RandomForestClassificationModel = {
    val model = RandomForestClassificationModel.load(path)
    model
  }

}

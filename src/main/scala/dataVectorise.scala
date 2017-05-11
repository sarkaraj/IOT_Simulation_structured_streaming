package src.main.scala

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._

/**
  * Created by rajsarka on 5/11/2017.
  */
object dataVectorise {
  def defineSchema(): StructType = {
    val Schema = new StructType(Array(
      StructField(name = "datetime", dataType = TimestampType),
      StructField(name = "user_id", dataType = StringType),
      StructField(name = "latitude", dataType = DoubleType),
      StructField(name = "longitude", dataType = DoubleType),
      StructField(name = "pulse", dataType = FloatType),
      StructField(name = "temp", dataType = FloatType),
      StructField(name = "age", dataType = IntegerType),
      StructField(name = "bp_category", dataType = StringType),
      StructField(name = "label", dataType = DoubleType)
    ))

    Schema
  }

  def vectorizeData(userData : DataFrame) : DataFrame = {

    val data = userData.select("pulse", "temp", "age", "bp_category", "label")

    val assembler_1 = new VectorAssembler()
      .setInputCols(Array("pulse", "temp", "age"))
      .setOutputCol("feature_1")

    val normalizeData = new StandardScaler()
      .setInputCol(assembler_1.getOutputCol)
      .setOutputCol("normalized_feature_1")

    val bp_cat_index = new StringIndexer()
      .setInputCol("bp_category")
      .setOutputCol("bp_cat_label")

    val assembler_2 = new VectorAssembler()
      .setInputCols(Array(assembler_1.getOutputCol, bp_cat_index.getOutputCol))
      .setOutputCol("features")

    val pipeline_1 = new Pipeline()
      .setStages(Array(assembler_1, normalizeData, bp_cat_index, assembler_2))

    val pipeLineModel_1 = pipeline_1.fit(data)

    val scaledData = pipeLineModel_1.transform(data)
      .select("label", "features")

    scaledData
  }

}

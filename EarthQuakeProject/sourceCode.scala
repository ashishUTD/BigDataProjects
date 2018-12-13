import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import java.io.PrintWriter
import java.io.File
import java.util
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer}
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer}
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.Normalizer

object project {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("project").setMaster("local"))

    val sqlContext = new SQLContext(sc)
    val spark = sqlContext.sparkSession

    import spark.implicits._

    // Reading the data
    var earthquake = spark.read.option("header", "true").option("inferSchema", "true").csv(args(0))

    // Filtering earthquake with magnitude > 2.5
    val new_df = earthquake.filter(($"label" > 2.5))

    // Drop null values
    var cleanEarthquake = new_df.na.drop()

    // Filtering data for type equals to "eq", i.e. earthquake
    cleanEarthquake = cleanEarthquake.filter($"type" === "eq").drop("type")
    cleanEarthquake = cleanEarthquake.drop("locationSource")
    cleanEarthquake = cleanEarthquake.drop("magSource")
    cleanEarthquake = cleanEarthquake.drop("status")
    cleanEarthquake = cleanEarthquake.drop("magNst", "nst")
    cleanEarthquake = cleanEarthquake.drop("time", "updated")
    cleanEarthquake = cleanEarthquake.drop("net")
    cleanEarthquake = cleanEarthquake.drop("place")
    cleanEarthquake = cleanEarthquake.drop("id")
    cleanEarthquake = cleanEarthquake.drop("nst")


    // Renaming magnitude to label
    val df = cleanEarthquake.withColumnRenamed( "mag", "label")

    // String indexer to convert String to Integer
    val indexer = new StringIndexer().setInputCol("magType").setOutputCol("newMagType").setHandleInvalid("skip")

    // Vector Assembler to generate features
    val assembler = new VectorAssembler()
    assembler.setInputCols(Array("latitude","longitude","depth","gap","dmin","rms","horizontalError","depthError","magError","newMagType"))
    assembler.setOutputCol("features")

    // Normalizer to normalize the data
    val normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures")
    val scaler = new StandardScaler().setInputCol("normFeatures").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)

    // PCA to do dimentionality reduction
    val pca = new PCA().setInputCol("scaledFeatures").setOutputCol("pcaFeatures").setK(3)

    // Decision Tree Model
    val dt = new DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("label")

    // Random Forest Model
    val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features")

    // Gradient Boosting Model
    val gbt = new GBTRegressor().setLabelCol("label").setFeaturesCol("features")

    // Linear Regression Model
    val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features")

    // Spliting the data into train and test 
    val Array(training, test) = df.randomSplit(Array(0.8, 0.2))

    // Creating a pipeline 
    var pipeline = new Pipeline().setStages(Array(indexer, assembler, normalizer, scaler, pca, dt))

    // Param Grid Builder to test different parameters
    var paramGrid = new ParamGridBuilder().addGrid(dt.maxDepth, Array(10, 20, 30))build()

    // Cross Validator
    var cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5)
    var cvModel = cv.fit(training)

    // Applying the cv model
    var prediction = cvModel.transform(test).select("scaledFeatures", "label", "prediction")

    // Different metrics calculation
    println("Decision Tree -- ")
    var eval=List("mse","rmse","mae","r2");
    for(e <-eval){
      var evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName(e)
      var se = evaluator.evaluate(prediction)
      println(s"Metric evalution = $e on test data = $se")
    }


    // Creating a pipeline 
    val pipeline_lr = new Pipeline().setStages(Array(indexer, assembler, normalizer, scaler, pca, lr))
    // Param Grid Builder to test different parameters
    val paramGrid_lr = new ParamGridBuilder().addGrid(lr.maxIter, Array(100, 200, 300)).addGrid(lr.regParam, Array(0.01, 0.1, 0.2))build()
    // Cross Validator
    val cv_lr = new CrossValidator().setEstimator(pipeline_lr).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid_lr).setNumFolds(5)
    // Applying the cv model
    val cvModel_lr = cv_lr.fit(training)
    val prediction_lr = cvModel_lr.transform(test).select("scaledFeatures", "label", "prediction")

    println("Linear Regression -- ")
    for(e <-eval){
      val evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName(e)
      val se = evaluator.evaluate(prediction_lr)
      println(s"Metric evalution = $e on test data = $se")
    }


    // Creating a pipeline 
    val pipeline_rf = new Pipeline().setStages(Array(indexer, assembler, normalizer, scaler, pca, rf))
    // Param Grid Builder to test different parameters
    val paramGrid_rf = new ParamGridBuilder().addGrid(rf.numTrees, Array(10, 15, 20)).addGrid(rf.maxDepth, Array(10, 15, 20))build()
    // Cross Validator
    val cv_rf = new CrossValidator().setEstimator(pipeline_rf).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid_rf).setNumFolds(5)
    // Applying the cv model
    val cvModel_rf = cv_rf.fit(training)
    val prediction_rf = cvModel_rf.transform(test).select("scaledFeatures", "label", "prediction")

    println("Random Forest -- ")
    for(e <-eval){
      val evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName(e)
      val se = evaluator.evaluate(prediction_rf)
      println(s"Metric evalution = $e on test data = $se")
    }


    // Creating a pipeline 
    val pipeline_gbt = new Pipeline().setStages(Array(indexer, assembler, gbt))
    // Param Grid Builder to test different parameters
    val paramGrid_gbt = new ParamGridBuilder().addGrid(gbt.maxDepth, Array(10, 20, 30))build()
    // Cross Validator
    val cv_gbt = new CrossValidator().setEstimator(pipeline_gbt).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid_gbt).setNumFolds(3)
    // Applying the cv model
    val cvModel_gbt = cv_gbt.fit(training)
    val prediction_gbt = cvModel_gbt.transform(test).select("features", "label", "prediction")

    println("Gradient Boosting -- ")
    for(e <-eval){
      val evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName(e)
      val se = evaluator.evaluate(prediction_gbt)
      println(s"Metric evalution = $e on test data = $se")
    }
  }
}

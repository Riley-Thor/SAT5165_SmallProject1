import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Bucketizer
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def main():
    # --- 1. Initialize Spark Session ---
    # This appName will show up in the Spark UI
    # The master URL points to our cluster.
    spark = SparkSession.builder \
        .appName("FlightDelayAnalysis") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    print("Spark Session created successfully.")
    start_time = time.time()

    # --- 2. Data Loading and Cleaning ---
    # Load the dataset. Spark infers the schema.
    data_path = "flights.csv" 
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Filter out cancelled and diverted flights as per the proposal
    df_cleaned = df.filter((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0))

    # Drop rows with missing values in the target variable
    df_cleaned = df_cleaned.na.drop(subset=["ARRIVAL_DELAY"])

    # Cast ARRIVAL_DELAY to integer for the model
    df_cleaned = df_cleaned.withColumn("ARRIVAL_DELAY", F.col("ARRIVAL_DELAY").cast("integer"))

    # 1 = Delayed (> 15 mins), 0 = On-Time
    df_cleaned = df_cleaned.withColumn("label", F.when(F.col("ARRIVAL_DELAY") > 15, 1).otherwise(0))

    # --- 3. Feature Engineering & Selection ---
    
    # Select the columns we need for the analysis
    features_df = df_cleaned.select(
        "MONTH", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY",
        "DISTANCE", "ARRIVAL_DELAY", "label"
    )

    # Create the TIME_OF_DAY feature by binning SCHEDULED_DEPARTURE (HHMM format)
    # Bins: [0, 600) -> Night (0), [600, 1200) -> Morning (1), [1200, 1800) -> Afternoon (2), [1800, 2400] -> Evening (3)
    bucketizer = Bucketizer(splits=[0, 600, 1200, 1800, 2400], 
                            inputCol="SCHEDULED_DEPARTURE", 
                            outputCol="TIME_OF_DAY_BIN")
    features_df = bucketizer.transform(features_df)

    # --- 4. Statistical Analysis (Correlation) ---
    print("\n--- Correlation Analysis ---")
    
    # Assemble numerical features for correlation
    corr_assembler = VectorAssembler(
        inputCols=["DEPARTURE_DELAY", "DISTANCE", "ARRIVAL_DELAY"],
        outputCol="corr_features"
    )
    corr_df = corr_assembler.transform(features_df).select("corr_features")

    # Compute Pearson correlation matrix
    matrix = Correlation.corr(corr_df, "corr_features").head()
    corr_matrix = matrix[0].toArray()
    
    print("Correlation matrix for [DEPARTURE_DELAY, DISTANCE, ARRIVAL_DELAY]:")
    print(corr_matrix)
    print("----------------------------\n")

    # --- 5. Data Preprocessing Pipeline ---
    
    # Identify categorical and numerical features for the model
    categorical_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "TIME_OF_DAY_BIN"]
    numerical_cols = ["MONTH", "DAY_OF_WEEK", "DEPARTURE_DELAY", "DISTANCE"]

    # Create StringIndexer stages for all categorical columns
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="skip")
        for col in categorical_cols
    ]

    # Create VectorAssembler to combine all features
    assembler_inputs = [col + "_index" for col in categorical_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # --- 6. Linear Regression Model ---
    
    # Define the GBT model
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10, maxBins=650)

    # Combine all stages into a single pipeline
    pipeline = Pipeline(stages=indexers + [assembler, gbt])

    # Split the data into training and testing sets
    (training_data, test_data) = features_df.randomSplit([0.8, 0.2], seed=42)

    # --- 7. Train and Evaluate Model ---
    print("Starting model training...")
    
    # Train the model
    pipeline_model = pipeline.fit(training_data)

    # Make predictions on the test set
    predictions = pipeline_model.transform(test_data)

    print("Model training and prediction complete.")

    # Evaluate the model
    print("\n--- Model Evaluation ---")
   
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator_auc.evaluate(predictions)
    print(f"Area Under ROC (AUC): {auc}")

    evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    
    accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
    precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
    recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("------------------------\n")

    # --- 8. Performance Measurement ---
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    spark.stop()

if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

INPUT_PATH = "/mnt/bigdata/coen542-project/data/processed/ml_binary"
MODEL_PATH = "/mnt/bigdata/coen542-project/models/random_forest_binary"

SPARK_TMP = "/mnt/bigdata/coen542-project/spark-tmp"


spark = (
    SparkSession.builder
    .appName("COEN542-RandomForest-Binary")
    .master("local[4]")
    .config("spark.driver.memory", "2g")
    .config("spark.local.dir", SPARK_TMP)
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("[INFO] Reading ML dataset...")
df = spark.read.parquet(INPUT_PATH)

print(f"[INFO] Total rows: {df.count()}")


# ---------------------------------------------------------
# 1. Identify feature columns
# ---------------------------------------------------------

feature_columns = [
    c for c in df.columns
    if c != "binary_label"
]

print(f"[INFO] Number of features: {len(feature_columns)}")


# ---------------------------------------------------------
# 2. Assemble features into Spark ML vector
# ---------------------------------------------------------

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

df_vector = assembler.transform(df).select(
    "features",
    col("binary_label").alias("label")
)


# ---------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------

train_df, test_df = df_vector.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("[INFO] 80/20 train/test split created.")


# ---------------------------------------------------------
# 4. Random Forest classifier
# ---------------------------------------------------------

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,
    maxDepth=12,
    seed=42
)

print("[INFO] Training Random Forest...")
model = rf.fit(train_df)

print("[INFO] Model training completed.")


# ---------------------------------------------------------
# 5. Generate predictions
# ---------------------------------------------------------

predictions = model.transform(test_df)

print("[INFO] Predictions generated.")


# ---------------------------------------------------------
# 6. Evaluate model
# ---------------------------------------------------------

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)


accuracy = accuracy_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)


print()
print("=" * 60)
print("RANDOM FOREST BINARY CLASSIFICATION RESULTS")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("=" * 60)


# ---------------------------------------------------------
# 7. Confusion matrix
# ---------------------------------------------------------

print()
print("Confusion Matrix:")
print("Rows = actual class")
print("Columns = predicted class")

confusion_matrix = (
    predictions
    .groupBy("label", "prediction")
    .count()
    .orderBy("label", "prediction")
)

confusion_matrix.show(truncate=False)


# ---------------------------------------------------------
# 8. Class-specific metrics
# ---------------------------------------------------------

print()
print("Prediction distribution:")

predictions.groupBy(
    "label",
    "prediction"
).count().orderBy(
    "label",
    "prediction"
).show()


# ---------------------------------------------------------
# 9. Feature importance
# ---------------------------------------------------------

print()
print("Top 15 Feature Importances:")

feature_importance = list(
    zip(feature_columns, model.featureImportances.toArray())
)

feature_importance.sort(
    key=lambda x: x[1],
    reverse=True
)

for feature, importance in feature_importance[:15]:
    print(f"{feature:40s} {importance:.6f}")


# ---------------------------------------------------------
# 10. Save model
# ---------------------------------------------------------

print()
print(f"[INFO] Saving model to: {MODEL_PATH}")

model.write().overwrite().save(MODEL_PATH)

print("[INFO] Model saved successfully.")


spark.stop()

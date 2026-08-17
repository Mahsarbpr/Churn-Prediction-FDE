resource "aws_iam_policy" "churn_service_s3_read" {
  name        = "churn-prediction-s3-read"
  description = "Allow the churn prediction service to read raw events and model artifacts"

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Action = [
          "s3:GetObject"
        ]

        Resource = [
          "${aws_s3_bucket.churn_data_lake.arn}/raw/*",
          "${aws_s3_bucket.churn_data_lake.arn}/models/*"
        ]
      }
    ]
  })
}
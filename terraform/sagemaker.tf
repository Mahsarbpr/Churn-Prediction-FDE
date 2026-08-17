resource "aws_iam_role" "sagemaker_training" {
  name = "churn-prediction-sagemaker-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "sagemaker.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_policy" "sagemaker_training_data" {
  name        = "churn-prediction-sagemaker-training-data"
  description = "Allow SageMaker training jobs to read training data and write model artifacts"

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Action = [
          "s3:ListBucket"
        ]

        Resource = [
          aws_s3_bucket.churn_data_lake.arn
        ]

        Condition = {
          StringLike = {
            "s3:prefix" = [
              "training/*"
            ]
          }
        }
      },
      {
        Effect = "Allow"

        Action = [
          "s3:GetObject"
        ]

        Resource = [
          "${aws_s3_bucket.churn_data_lake.arn}/training/*"
        ]
      },
      {
        Effect = "Allow"

        Action = [
          "s3:PutObject"
        ]

        Resource = [
          "${aws_s3_bucket.churn_data_lake.arn}/models/*"
        ]
      },
      {
        Effect = "Allow"

        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_training_data" {
  role       = aws_iam_role.sagemaker_training.name
  policy_arn = aws_iam_policy.sagemaker_training_data.arn
}

resource "aws_iam_role_policy_attachment" "sagemaker_ecr_read" {
  role       = aws_iam_role.sagemaker_training.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}
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

resource "aws_iam_role" "churn_service" {
  name = "churn-prediction-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "pods.eks.amazonaws.com"
        }

        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "churn_service_s3_read" {
  role       = aws_iam_role.churn_service.name
  policy_arn = aws_iam_policy.churn_service_s3_read.arn
}

resource "aws_eks_pod_identity_association" "churn_service" {
  cluster_name    = aws_eks_cluster.churn.name
  namespace       = "default"
  service_account = "churn-service"
  role_arn        = aws_iam_role.churn_service.arn

  depends_on = [
    aws_eks_addon.pod_identity_agent,
  ]
}
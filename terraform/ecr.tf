resource "aws_ecr_repository" "churn_service" {
  name                 = "churn-prediction-service"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
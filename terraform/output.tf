output "ecr_repository_url" {
  value = aws_ecr_repository.churn_service.repository_url
}

output "s3_data_lake_bucket_name" {
  value = aws_s3_bucket.churn_data_lake.bucket
}
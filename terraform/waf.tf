resource "aws_wafv2_web_acl" "churn_service" {
  name  = "churn-prediction-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "rate-limit"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 100
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "churn-prediction-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "churn-prediction-waf"
    sampled_requests_enabled   = true
  }
}

resource "aws_wafv2_web_acl_association" "churn_service" {
  resource_arn = "arn:aws:elasticloadbalancing:us-east-1:239302213769:loadbalancer/app/k8s-default-churnser-c3b7b9912a/02d114144aaea300"
  web_acl_arn  = aws_wafv2_web_acl.churn_service.arn
}
resource "aws_cloudwatch_dashboard" "churn_prediction" {
  dashboard_name = "churn-prediction"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          title  = "Prediction Latency"
          region = "us-east-1"

          metrics = [
            [
              "ChurnPrediction",
              "churn_prediction.request_latency_ms",
              "route",
              "/predict",
            ]
          ]

          stat   = "Average"
          period = 60
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          title  = "Prediction Errors"
          region = "us-east-1"

          metrics = [
            [
              "ChurnPrediction",
              "churn_prediction.errors",
            ]
          ]

          stat   = "Sum"
          period = 60
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 24
        height = 6

        properties = {
          title  = "Churn Score Distribution"
          region = "us-east-1"

          metrics = [
            [
              "ChurnPrediction",
              "churn_prediction.score",
            ]
          ]

          stat   = "Average"
          period = 60
        }
      }
    ]
  })
}
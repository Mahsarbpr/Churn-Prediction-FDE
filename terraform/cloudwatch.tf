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
              "OTelLib",
              "churn-prediction-service"
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
              "error_type",
              "customer_not_found",
              "OTelLib",
              "churn-prediction-service"
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
          title  = "Average Churn Score"
          region = "us-east-1"

          metrics = [
            [
              "ChurnPrediction",
              "churn_prediction.score",
              "OTelLib",
              "churn-prediction-service"
            ]
          ]

          stat   = "Average"
          period = 60
        }
      }
    ]
  })
}
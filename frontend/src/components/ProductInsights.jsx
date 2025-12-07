import React from 'react'
import './ProductInsights.css'

function ProductInsights({ insights }) {
  const getSentimentColor = (label) => {
    switch (label.toLowerCase()) {
      case 'positive':
        return '#10b981'
      case 'negative':
        return '#ef4444'
      default:
        return '#6b7280'
    }
  }

  const getSentimentEmoji = (label) => {
    switch (label.toLowerCase()) {
      case 'positive':
        return '😊'
      case 'negative':
        return '😞'
      default:
        return '😐'
    }
  }

  return (
    <div className="insights-container">
      <div className="insights-header">
        <h2>{insights.product_name}</h2>
      </div>

      <div className="insights-grid">
        {/* Statistics Card */}
        <div className="insight-card statistics-card">
          <h3>📊 Statistics</h3>
          <div className="statistics-grid">
            <div className="stat-item">
              <span className="stat-label">Total Reviews</span>
              <span className="stat-value">{insights.total_reviews.toLocaleString()}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Average Rating</span>
              <span className="stat-value">
                {insights.average_rating.toFixed(2)} / 5.0
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Sentiment</span>
              <span 
                className="stat-value sentiment-badge"
                style={{ color: getSentimentColor(insights.sentiment_label) }}
              >
                {getSentimentEmoji(insights.sentiment_label)} {insights.sentiment_label}
                <span className="sentiment-score">
                  ({insights.average_sentiment > 0 ? '+' : ''}{insights.average_sentiment.toFixed(3)})
                </span>
              </span>
            </div>
          </div>
        </div>

        {/* Positive Keywords Card */}
        <div className="insight-card positive-card">
          <h3>✅ Top Positive Keywords</h3>
          {insights.top_positive_keywords && insights.top_positive_keywords.length > 0 ? (
            <ul className="keyword-list">
              {insights.top_positive_keywords.map((keyword, index) => (
                <li key={index} className="keyword-item">
                  <span className="keyword-rank">{index + 1}.</span>
                  <span className="keyword-phrase">{keyword.phrase}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-keywords">No positive keywords found</p>
          )}
        </div>

        {/* Negative Keywords Card */}
        <div className="insight-card negative-card">
          <h3>❌ Top Negative Keywords</h3>
          {insights.top_negative_keywords && insights.top_negative_keywords.length > 0 ? (
            <ul className="keyword-list">
              {insights.top_negative_keywords.map((keyword, index) => (
                <li key={index} className="keyword-item">
                  <span className="keyword-rank">{index + 1}.</span>
                  <span className="keyword-phrase">{keyword.phrase}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-keywords">No negative keywords found</p>
          )}
        </div>

        {/* Summary Card */}
        <div className="insight-card summary-card">
          <h3>📝 Summary</h3>
          <p className="summary-text">{insights.summary}</p>
        </div>
      </div>
    </div>
  )
}

export default ProductInsights


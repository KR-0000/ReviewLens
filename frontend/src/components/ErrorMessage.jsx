import React from 'react'
import './ErrorMessage.css'

function ErrorMessage({ message, onDismiss }) {
  return (
    <div className="error-container">
      <div className="error-content">
        <span className="error-icon">⚠️</span>
        <span className="error-message">{message}</span>
        {onDismiss && (
          <button className="error-dismiss" onClick={onDismiss}>
            ×
          </button>
        )}
      </div>
    </div>
  )
}

export default ErrorMessage


import React, { useState } from 'react'
import SearchBar from './components/SearchBar'
import ProductInsights from './components/ProductInsights'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorMessage from './components/ErrorMessage'
import './App.css'

function App() {
  const [insights, setInsights] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSearch = async (productName) => {
    if (!productName || !productName.trim()) {
      setError('Please enter a product name')
      return
    }

    setLoading(true)
    setError(null)
    setInsights(null)

    try {
      const response = await fetch(`/api/products/${encodeURIComponent(productName)}/insights`)
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`No insights found for "${productName}". Please try a different product name.`)
        }
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch product insights')
      }
      const data = await response.json()
      setInsights(data)
    } catch (err) {
      setError(err.message || 'An error occurred while fetching product insights')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>ReviewLens</h1>
      </header>

      <main className="app-main">
        <SearchBar onSearch={handleSearch} disabled={loading} />
        
        {loading && <LoadingSpinner />}
        
        {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}
        
        {insights && !loading && <ProductInsights insights={insights} />}
      </main>

    </div>
  )
}

export default App


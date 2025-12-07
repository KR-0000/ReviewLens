import React, { useState } from 'react'
import './SearchBar.css'

function SearchBar({ onSearch, disabled }) {
  const [searchTerm, setSearchTerm] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(false)

  const handleInputChange = async (e) => {
    const value = e.target.value
    setSearchTerm(value)

    if (value.length >= 2) {
      try {
        const response = await fetch(`/api/products/search?q=${encodeURIComponent(value)}`)
        if (response.ok) {
          const data = await response.json()
          setSuggestions(data.products.slice(0, 5))
          setShowSuggestions(true)
        }
      } catch (err) {
        console.error('Error fetching suggestions:', err)
      }
    } else {
      setSuggestions([])
      setShowSuggestions(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (searchTerm.trim() && !disabled) {
      onSearch(searchTerm.trim())
      setShowSuggestions(false)
    }
  }

  const handleSuggestionClick = (productName) => {
    setSearchTerm(productName)
    setShowSuggestions(false)
    onSearch(productName)
  }

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit} className="search-form">
        <div className="search-input-wrapper">
          <input
            type="text"
            value={searchTerm}
            onChange={handleInputChange}
            placeholder="Search for a product..."
            className="search-input"
            disabled={disabled}
            autoComplete="off"
          />
          <button
            type="submit"
            className="search-button"
            disabled={disabled || !searchTerm.trim()}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="11" cy="11" r="8"></circle>
              <path d="m21 21-4.35-4.35"></path>
            </svg>
          </button>
        </div>
        
        {showSuggestions && suggestions.length > 0 && (
          <div className="suggestions-dropdown">
            {suggestions.map((product, index) => (
              <div
                key={index}
                className="suggestion-item"
                onClick={() => handleSuggestionClick(product)}
              >
                {product}
              </div>
            ))}
          </div>
        )}
      </form>
    </div>
  )
}

export default SearchBar


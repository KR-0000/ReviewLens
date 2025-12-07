"""
Process reviews: compute sentiment, keywords, and summaries.
This script analyzes review text and stores results back in SQLite.
"""

import sqlite3
import re
from collections import Counter
from typing import List, Dict, Tuple, Set
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from transformers import pipeline
import torch

# Import spaCy
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class ReviewProcessor:
    """Process reviews for sentiment, keywords, and summaries."""
    
    def __init__(self, db_path='reviews.db', use_gpu=False):
        self.db_path = db_path
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        # Add common words to stopwords
        self.stop_words.update(['product', 'item', 'amazon', 'purchase', 'buy', 'bought'])
        
        # Initialize spaCy for phrase extraction
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                print("Loading spaCy model...")
                self.nlp = spacy.load("en_core_web_sm")
                print("spaCy model loaded successfully.")
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found.")
                print("Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            print("Warning: spaCy not available. Using fallback phrase extraction.")
        
        # Initialize summarization model (use CPU by default, can use GPU if available)
        print("Loading summarization model...")
        device = 0 if use_gpu and torch.cuda.is_available() else -1
        self.summarizer = None
        
        # Try smaller models first for better compatibility
        models_to_try = [
            ("t5-small", "t5-small"),
            ("facebook/bart-large-cnn", "facebook/bart-large-cnn"),
            ("google/pegasus-xsum", "google/pegasus-xsum")
        ]
        
        for model_name, tokenizer_name in models_to_try:
            try:
                print(f"Attempting to load {model_name}...")
                self.summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    tokenizer=tokenizer_name,
                    device=device,
                    model_kwargs={"torch_dtype": torch.float16 if device >= 0 else torch.float32} if "bart" in model_name.lower() else {}
                )
                print(f"Successfully loaded {model_name}.")
                break
            except Exception as e:
                print(f"Could not load {model_name}: {e}")
                continue
        
        if self.summarizer is None:
            print("Warning: Could not load any transformer model.")
            print("Falling back to extractive summarization.")
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        return text.strip()
    
    def compute_sentiment(self, text: str) -> Tuple[float, str]:
        """Compute sentiment score using VADER."""
        if not text:
            return 0.0, 'neutral'
        
        text = self.clean_text(text)
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return compound, label
    
    def _extract_phrases_spacy(self, text: str) -> List[Tuple[str, str]]:
        """Extract noun chunks and verb phrases using spaCy. Returns (phrase, sentence) tuples."""
        if not self.nlp or not text:
            return []
        
        doc = self.nlp(text)
        phrases = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Extract noun chunks
            for chunk in sent.noun_chunks:
                # Filter out very short or stopword-only chunks
                chunk_text = chunk.text.lower().strip()
                words = [w.text.lower() for w in chunk if not w.is_stop and w.is_alpha and len(w.text) > 2]
                if len(words) >= 1 and len(chunk_text.split()) <= 4:  # 2-4 word phrases
                    # Remove internal repetition (e.g., "great great" -> "great")
                    words_clean = []
                    prev_word = None
                    for word in words:
                        if word != prev_word:
                            words_clean.append(word)
                        prev_word = word
                    if words_clean:
                        phrase = " ".join(words_clean)
                        if len(phrase.split()) >= 1:
                            phrases.append((phrase, sent_text))
            
            # Extract verb phrases (verb + object/direct object)
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp"]:
                    # Get verb and its direct object
                    verb_phrase_parts = [token.text.lower()]
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj", "attr", "acomp"]:
                            # Get the noun phrase
                            obj_text = child.text.lower()
                            if child.pos_ in ["NOUN", "PROPN"]:
                                verb_phrase_parts.append(obj_text)
                            elif child.pos_ == "ADJ":
                                verb_phrase_parts.append(obj_text)
                    if len(verb_phrase_parts) >= 2 and len(verb_phrase_parts) <= 4:
                        phrase = " ".join(verb_phrase_parts)
                        phrases.append((phrase, sent_text))
        
        return phrases
    
    def _extract_phrases_fallback(self, text: str) -> List[Tuple[str, str]]:
        """Fallback phrase extraction using NLTK when spaCy is unavailable."""
        if not text:
            return []
        
        sentences = sent_tokenize(text)
        phrases = []
        
        for sent in sentences:
            tokens = word_tokenize(sent.lower())
            tokens = [w for w in tokens if w.isalpha() and w not in self.stop_words and len(w) > 2]
            
            # Extract bigrams and trigrams
            for i in range(len(tokens) - 1):
                if i + 1 < len(tokens):
                    phrase = f"{tokens[i]} {tokens[i+1]}"
                    # Remove repetition
                    words = phrase.split()
                    words_clean = []
                    prev = None
                    for word in words:
                        if word != prev:
                            words_clean.append(word)
                        prev = word
                    if words_clean:
                        phrases.append((" ".join(words_clean), sent))
            
            for i in range(len(tokens) - 2):
                if i + 2 < len(tokens):
                    phrase = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                    words = phrase.split()
                    words_clean = []
                    prev = None
                    for word in words:
                        if word != prev:
                            words_clean.append(word)
                        prev = word
                    if words_clean:
                        phrases.append((" ".join(words_clean), sent))
        
        return phrases
    
    def extract_keywords(self, texts: List[str], sentiment_target: str, product_name: str = "", top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Extract keywords with phrase-level sentiment scoring using spaCy for proper phrase extraction.
        sentiment_target: 'positive' or 'negative'
        Returns: List of (phrase, sentiment_score) tuples
        """
        if not texts:
            return []
        
        # Extract product title terms for filtering (normalize product name)
        product_terms = set()
        if product_name:
            # Normalize: remove punctuation, extra spaces, convert to lowercase
            product_name_clean = re.sub(r'[^\w\s]', ' ', product_name.lower())
            product_name_clean = re.sub(r'\s+', ' ', product_name_clean).strip()
            product_words = product_name_clean.split()
            product_words = [w for w in product_words if len(w) > 2]
            # Create unigrams, bigrams, and trigrams from product name
            for i in range(len(product_words)):
                product_terms.add(product_words[i])
                if i < len(product_words) - 1:
                    product_terms.add(f"{product_words[i]} {product_words[i+1]}")
                if i < len(product_words) - 2:
                    product_terms.add(f"{product_words[i]} {product_words[i+1]} {product_words[i+2]}")
        
        # Generic/neutral words to filter out
        generic_words = {
            'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third',
            'time', 'times', 'day', 'days', 'week', 'weeks', 'month', 'months',
            'year', 'years', 'hour', 'hours', 'minute', 'minutes',
            'thing', 'things', 'stuff', 'way', 'ways', 'part', 'parts',
            'kids', 'kid', 'child', 'children', 'people', 'person',
            'use', 'uses', 'used', 'using', 'work', 'works', 'working',
            'get', 'gets', 'got', 'make', 'makes', 'made', 'take', 'takes', 'took',
            'fire', 'purchased', 'purchase', 'bought', 'buy'
        }
        
        # Strongly sentiment-bearing single words (allowed even if single word)
        strong_sentiment_words = {
            'love', 'hate', 'excellent', 'terrible', 'amazing', 'awful', 'perfect', 'horrible',
            'great', 'bad', 'good', 'worst', 'best', 'fantastic', 'disappointing', 'disappointed',
            'broken', 'dead', 'useless', 'wonderful', 'terrible', 'horrible',
            'satisfied', 'unsatisfied', 'happy', 'unhappy', 'pleased', 'displeased',
            'recommend', 'recommended', 'avoid', 'avoided'
        }
        
        # Collect phrases from all texts
        phrase_candidates = {}  # phrase -> (count, sentences, original_phrases)
        
        for text in texts:
            if not text:
                continue
            
            text_clean = self.clean_text(text)
            
            # Extract phrases using spaCy or fallback
            if self.nlp:
                phrases = self._extract_phrases_spacy(text_clean)
            else:
                phrases = self._extract_phrases_fallback(text_clean)
            
            for phrase, sentence in phrases:
                phrase_lower = phrase.lower().strip()
                
                # Skip if phrase is too short or too long
                words = phrase_lower.split()
                if len(words) < 1 or len(words) > 4:
                    continue
                
                # Skip single words unless they're strongly sentiment-bearing
                if len(words) == 1 and words[0] not in strong_sentiment_words:
                    continue
                
                # Remove internal repetition (e.g., "great great" -> "great")
                words_clean = []
                prev_word = None
                for word in words:
                    if word != prev_word:
                        words_clean.append(word)
                    prev_word = word
                
                if not words_clean:
                    continue
                
                phrase_clean = " ".join(words_clean)
                
                # Initialize if new phrase
                if phrase_clean not in phrase_candidates:
                    phrase_candidates[phrase_clean] = [0, [], []]
                
                phrase_candidates[phrase_clean][0] += 1
                phrase_candidates[phrase_clean][1].append(sentence)
                phrase_candidates[phrase_clean][2].append(phrase)  # Store original for reference
        
        # Score each phrase by its own sentiment
        scored_phrases = []
        
        for phrase, (count, sentences, originals) in phrase_candidates.items():
            if count < 2:  # Minimum frequency threshold
                continue
            
            # Filter out product title terms (unless they have sentiment modifiers)
            phrase_words = set(phrase.split())
            if phrase_words.intersection(product_terms):
                # Check if sentence contains sentiment words
                has_sentiment_modifier = False
                for sentence in sentences[:3]:  # Check first few occurrences
                    sentence_sentiment = self.sia.polarity_scores(sentence)
                    # Also check for explicit sentiment words in sentence
                    sentence_lower = sentence.lower()
                    sentiment_indicators = ['love', 'hate', 'great', 'terrible', 'excellent', 'awful', 
                                           'amazing', 'horrible', 'perfect', 'disappointing', 'fantastic']
                    if abs(sentence_sentiment['compound']) > 0.3 or any(ind in sentence_lower for ind in sentiment_indicators):
                        has_sentiment_modifier = True
                        break
                if not has_sentiment_modifier:
                    continue
            
            # Filter out generic words (especially for negative sentiment)
            if sentiment_target == 'negative':
                if len(phrase_words) == 1 and phrase in generic_words:
                    continue
                # Also filter if phrase is mostly generic words
                generic_ratio = len(phrase_words.intersection(generic_words)) / len(phrase_words) if phrase_words else 0
                if generic_ratio > 0.5 and len(phrase_words) > 1:
                    continue
            
            # Score phrase sentiment using the sentences it appears in
            phrase_sentiments = []
            for sentence in sentences[:10]:  # Use up to 10 sentences for scoring
                # Score the phrase itself
                phrase_sentiment = self.sia.polarity_scores(phrase)
                # Also score the sentence context
                sentence_sentiment = self.sia.polarity_scores(sentence)
                # Weight: 60% phrase itself, 40% sentence context
                combined_sentiment = 0.6 * phrase_sentiment['compound'] + 0.4 * sentence_sentiment['compound']
                phrase_sentiments.append(combined_sentiment)
            
            if not phrase_sentiments:
                continue
            
            avg_sentiment = sum(phrase_sentiments) / len(phrase_sentiments)
            
            # Filter by sentiment target
            if sentiment_target == 'positive' and avg_sentiment < 0.1:
                continue
            if sentiment_target == 'negative' and avg_sentiment > -0.1:
                continue
            
            # Prefer multi-word phrases: boost score for longer phrases
            word_count = len(phrase.split())
            length_bonus = 0.15 * (word_count - 1) if word_count > 1 else 0  # Bonus for multi-word phrases
            
            # Combine sentiment score with frequency and length
            # Weight: 65% sentiment strength, 20% frequency (normalized), 15% length bonus
            freq_score = min(count / 8.0, 1.0)  # Normalize frequency
            sentiment_strength = abs(avg_sentiment)
            combined_score = (0.65 * sentiment_strength) + (0.20 * freq_score) + (0.15 * length_bonus)
            
            scored_phrases.append((phrase, combined_score, avg_sentiment, count))
        
        # Sort by combined score (sentiment + frequency + length)
        scored_phrases.sort(key=lambda x: x[1], reverse=True)
        
        # Remove overlapping keywords
        filtered = self._remove_overlapping_keywords_scored(scored_phrases)
        
        # Return top N with their sentiment scores
        return [(phrase, sentiment) for phrase, _, sentiment, _ in filtered[:top_n]]
    
    def _remove_overlapping_keywords_scored(self, scored_phrases: List[Tuple[str, float, float, int]]) -> List[Tuple[str, float, float, int]]:
        """Remove overlapping keywords from scored phrases."""
        if not scored_phrases:
            return []
        
        # Sort by score (highest first)
        sorted_phrases = sorted(scored_phrases, key=lambda x: x[1], reverse=True)
        
        seen_word_sets = {}
        filtered = []
        
        for phrase, combined_score, sentiment, count in sorted_phrases:
            phrase_lower = phrase.lower().strip()
            keyword_words = frozenset(phrase_lower.split())
            
            should_skip = False
            for seen_word_set, seen_phrase in seen_word_sets.items():
                if keyword_words == seen_word_set:
                    should_skip = True
                    break
                elif keyword_words.issubset(seen_word_set):
                    should_skip = True
                    break
            
            if not should_skip:
                filtered.append((phrase, combined_score, sentiment, count))
                seen_word_sets[keyword_words] = phrase_lower
        
        return filtered
    
    def _remove_overlapping_keywords(self, keywords: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Remove duplicate and overlapping keywords within the same list."""
        if not keywords:
            return []
        
        # First, remove exact duplicates (case-insensitive)
        seen_exact = set()
        unique_keywords = []
        for keyword, count in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower not in seen_exact:
                seen_exact.add(keyword_lower)
                unique_keywords.append((keyword, count))
        
        # Sort by length (longer first) and frequency (higher first)
        keywords_sorted = sorted(unique_keywords, key=lambda x: (len(x[0].split()), x[1]), reverse=True)
        
        seen_word_sets = {}  # Map from frozenset of words to the keyword string
        filtered = []
        
        for keyword, count in keywords_sorted:
            keyword_lower = keyword.lower().strip()
            keyword_words = frozenset(keyword_lower.split())
            
            # Check if this keyword overlaps with an already added keyword
            should_skip = False
            for seen_word_set, seen_keyword in seen_word_sets.items():
                # If same words (different order), skip
                if keyword_words == seen_word_set:
                    should_skip = True
                    break
                # If current keyword is a subset of a seen keyword, skip it
                elif keyword_words.issubset(seen_word_set):
                    should_skip = True
                    break
            
            if not should_skip:
                filtered.append((keyword, count))
                seen_word_sets[keyword_words] = keyword_lower
        
        return filtered
    
    def generate_insight_summary(self, texts: List[str], positive_keywords: List[Tuple[str, float]], 
                                  negative_keywords: List[Tuple[str, float]], 
                                  max_length: int = 200, min_length: int = 80) -> str:
        """Generate insight-focused summary using themes, not literal keywords."""
        if not texts:
            return ""
        
        # Extract themes from keywords (not literal phrases)
        pos_themes = self._extract_themes_from_keywords(positive_keywords)
        neg_themes = self._extract_themes_from_keywords(negative_keywords)
        
        # Get representative high-sentiment sentences
        representative_sentences = self._get_representative_sentences(texts, positive_keywords, negative_keywords)
        
        # Combine all texts
        combined_text = " ".join([self.clean_text(t) for t in texts if t])
        if not combined_text:
            return ""
        
        # Truncate if too long
        if len(combined_text) > 2000:
            sentences = sent_tokenize(combined_text)
            truncated = ""
            for sent in sentences:
                if len(truncated + sent) > 2000:
                    break
                truncated += sent + " "
            combined_text = truncated.strip()
        
        # Build structured input for summarization
        structured_input = self._build_structured_input(pos_themes, neg_themes, representative_sentences, combined_text)
        
        # Use transformer model for abstractive summarization
        if self.summarizer:
            try:
                # Generate summary from structured input
                summary = self.summarizer(
                    structured_input,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True
                )
                summary_text = summary[0]['summary_text']
                
                # Clean up any remaining keyword artifacts
                summary_text = self._clean_summary_text(summary_text, pos_themes, neg_themes)
                
                return summary_text
            except Exception as e:
                print(f"Warning: Transformer summarization failed: {e}")
                print("Falling back to theme-based summary.")
                return self._theme_based_summary(pos_themes, neg_themes, texts)
        else:
            return self._theme_based_summary(pos_themes, neg_themes, texts)
    
    def _extract_themes_from_keywords(self, keywords: List[Tuple[str, float]]) -> List[str]:
        """Extract themes from keywords, not literal phrases."""
        themes = []
        for phrase, score in keywords[:5]:
            # Extract meaningful words from phrase (remove filler)
            words = [w for w in phrase.split() if w not in self.stop_words and len(w) > 3]
            if words:
                # Create theme description
                if len(words) == 1:
                    theme = words[0]
                else:
                    # Use the most meaningful words
                    theme = " ".join(words[:2])
                themes.append(theme)
        return themes
    
    def _get_representative_sentences(self, texts: List[str], pos_keywords: List[Tuple[str, float]], 
                                     neg_keywords: List[Tuple[str, float]], num_sentences: int = 5) -> List[str]:
        """Get representative high-sentiment sentences from reviews."""
        all_sentences = []
        for text in texts[:20]:  # Sample from first 20 reviews
            if not text:
                continue
            text_clean = self.clean_text(text)
            sentences = sent_tokenize(text_clean)
            for sent in sentences:
                if len(sent.split()) >= 5:  # Meaningful sentences
                    sentiment = self.sia.polarity_scores(sent)
                    if abs(sentiment['compound']) > 0.3:  # Strong sentiment
                        all_sentences.append((sent, abs(sentiment['compound'])))
        
        # Sort by sentiment strength and return top sentences
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in all_sentences[:num_sentences]]
    
    def _build_structured_input(self, pos_themes: List[str], neg_themes: List[str], 
                               representative_sentences: List[str], combined_text: str) -> str:
        """Build structured input for summarization."""
        parts = []
        
        if pos_themes:
            parts.append(f"Key positive themes: {', '.join(pos_themes[:3])}.")
        
        if neg_themes:
            parts.append(f"Key concerns: {', '.join(neg_themes[:3])}.")
        
        if representative_sentences:
            parts.append("Representative review excerpts:")
            for sent in representative_sentences[:3]:
                parts.append(f"- {sent}")
        
        parts.append("\nAll reviews:")
        parts.append(combined_text)
        
        return " ".join(parts)
    
    def _clean_summary_text(self, summary: str, pos_themes: List[str], neg_themes: List[str]) -> str:
        """Clean summary text to remove keyword artifacts."""
        # Remove any literal keyword repetition
        summary_words = summary.split()
        cleaned_words = []
        prev_word = None
        for word in summary_words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
            prev_word = word.lower()
        
        summary = " ".join(cleaned_words)
        
        # Ensure summary doesn't literally repeat themes
        summary_lower = summary.lower()
        for theme in pos_themes + neg_themes:
            theme_lower = theme.lower()
            # If theme appears multiple times, it's likely an artifact
            if summary_lower.count(theme_lower) > 2:
                # Replace excessive occurrences
                summary = re.sub(rf'\b{re.escape(theme)}\b', '', summary, flags=re.IGNORECASE)
                summary = re.sub(r'\s+', ' ', summary).strip()
        
        return summary
    
    def _theme_based_summary(self, pos_themes: List[str], neg_themes: List[str], texts: List[str]) -> str:
        """Generate summary based on themes when transformer fails."""
        summary_parts = []
        
        # Build positive summary
        if pos_themes:
            if len(pos_themes) == 1:
                summary_parts.append(f"Reviewers consistently highlight {pos_themes[0]} as a positive aspect.")
            elif len(pos_themes) == 2:
                summary_parts.append(f"Reviewers consistently highlight {pos_themes[0]} and {pos_themes[1]} as positive aspects.")
            else:
                summary_parts.append(f"Reviewers consistently highlight {pos_themes[0]}, {pos_themes[1]}, and {pos_themes[2]} as positive aspects.")
        
        # Build negative summary
        if neg_themes:
            if summary_parts:
                summary_parts.append(" ")
            if len(neg_themes) == 1:
                summary_parts.append(f"However, common concerns relate to {neg_themes[0]}.")
            elif len(neg_themes) == 2:
                summary_parts.append(f"However, common concerns relate to {neg_themes[0]} and {neg_themes[1]}.")
            else:
                summary_parts.append(f"However, common concerns relate to {neg_themes[0]}, {neg_themes[1]}, and {neg_themes[2]}.")
        
        # Add overall sentiment
        if texts:
            all_text = " ".join([self.clean_text(t) for t in texts[:15] if t])
            if all_text:
                overall_sentiment = self.sia.polarity_scores(all_text)
                if overall_sentiment['compound'] > 0.3:
                    summary_parts.append("Overall, the product receives predominantly positive feedback.")
                elif overall_sentiment['compound'] < -0.3:
                    summary_parts.append("Overall, reviewers express significant concerns.")
                else:
                    summary_parts.append("Reviews present a mixed perspective with both positive and negative aspects.")
        
        return "".join(summary_parts)
    
    def generate_summary(self, texts: List[str], max_length: int = 150, min_length: int = 50) -> str:
        """Generate abstractive summary using transformer model (legacy method)."""
        if not texts:
            return ""
        
        # Combine all texts
        combined_text = " ".join([self.clean_text(t) for t in texts if t])
        if not combined_text:
            return ""
        
        # Truncate if too long (transformers have token limits)
        # Most models can handle ~1024 tokens, so we'll limit to ~2000 characters
        if len(combined_text) > 2000:
            # Try to truncate at sentence boundaries
            sentences = sent_tokenize(combined_text)
            truncated = ""
            for sent in sentences:
                if len(truncated + sent) > 2000:
                    break
                truncated += sent + " "
            combined_text = truncated.strip()
        
        # Use transformer model for abstractive summarization
        if self.summarizer:
            try:
                # Generate summary
                summary = self.summarizer(
                    combined_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True
                )
                return summary[0]['summary_text']
            except Exception as e:
                print(f"Warning: Transformer summarization failed: {e}")
                print("Falling back to extractive summarization.")
                return self._extractive_fallback(combined_text)
        else:
            return self._extractive_fallback(combined_text)
    
    def _extractive_fallback(self, text: str, num_sentences: int = 3) -> str:
        """Fallback extractive summary method."""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        # Simple scoring
        word_freq = Counter()
        for sent in sentences:
            words = word_tokenize(sent.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            word_freq.update(words)
        
        sentence_scores = {}
        for sent in sentences:
            words = word_tokenize(sent.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            score = sum(word_freq.get(w, 0) for w in words) / (len(words) + 1)
            sentence_scores[sent] = score
        
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences = [s[0] for s in top_sentences]
        
        summary_sentences = []
        for sent in sentences:
            if sent in top_sentences:
                summary_sentences.append(sent)
                if len(summary_sentences) >= num_sentences:
                    break
        
        return " ".join(summary_sentences)
    
    def process_all_reviews(self, batch_size: int = 100):
        """Process all reviews and update sentiment scores."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        print("Processing reviews for sentiment analysis...")
        
        # Get unprocessed reviews
        cursor.execute('SELECT rowid, reviews_text FROM reviews WHERE processed = 0')
        rows = cursor.fetchall()
        
        total = len(rows)
        print(f"Found {total} reviews to process")
        
        for i, (rowid, text) in enumerate(rows):
            if text:
                sentiment_score, sentiment_label = self.compute_sentiment(text)
                cursor.execute(
                    'UPDATE reviews SET sentiment_score = ?, sentiment_label = ?, processed = 1 WHERE rowid = ?',
                    (sentiment_score, sentiment_label, rowid)
                )
            
            if (i + 1) % batch_size == 0:
                conn.commit()
                print(f"Processed {i + 1}/{total} reviews...")
        
        conn.commit()
        print(f"Completed processing {total} reviews.")
        conn.close()
    
    def create_keywords_table(self):
        """Create table to store product keywords."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DROP TABLE IF EXISTS product_keywords')
        cursor.execute('''
            CREATE TABLE product_keywords (
                product_name TEXT PRIMARY KEY,
                positive_keywords TEXT,
                negative_keywords TEXT,
                summary TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_products(self):
        """Process products to extract keywords and summaries."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        print("\nProcessing products for keywords and summaries...")
        
        # Get all unique products
        cursor.execute('SELECT DISTINCT name FROM reviews WHERE name IS NOT NULL AND name != ""')
        products = cursor.fetchall()
        
        total_products = len(products)
        print(f"Found {total_products} unique products")
        
        self.create_keywords_table()
        
        for idx, (product_name,) in enumerate(products):
            # Get all reviews for this product
            cursor.execute(
                'SELECT reviews_text, sentiment_label FROM reviews WHERE name = ? AND reviews_text IS NOT NULL AND reviews_text != ""',
                (product_name,)
            )
            reviews = cursor.fetchall()
            
            if not reviews:
                continue
            
            texts = [r[0] for r in reviews]
            positive_texts = [r[0] for r in reviews if r[1] == 'positive']
            negative_texts = [r[0] for r in reviews if r[1] == 'negative']
            
            # Extract keywords with phrase-level sentiment scoring - top 5 only
            positive_keywords = self.extract_keywords(positive_texts, 'positive', product_name, top_n=5)
            negative_keywords = self.extract_keywords(negative_texts, 'negative', product_name, top_n=5)
            
            # Generate insight-focused summary incorporating keywords
            summary = self.generate_insight_summary(texts, positive_keywords, negative_keywords, max_length=200, min_length=80)
            
            # Store in database (format: phrase(sentiment_score))
            pos_keywords_str = ", ".join([f"{phrase}({score:.2f})" for phrase, score in positive_keywords])
            neg_keywords_str = ", ".join([f"{phrase}({score:.2f})" for phrase, score in negative_keywords])
            
            cursor.execute(
                'INSERT OR REPLACE INTO product_keywords (product_name, positive_keywords, negative_keywords, summary) VALUES (?, ?, ?, ?)',
                (product_name, pos_keywords_str, neg_keywords_str, summary)
            )
            
            if (idx + 1) % 100 == 0:
                conn.commit()
                print(f"Processed {idx + 1}/{total_products} products...")
        
        conn.commit()
        print(f"Completed processing {total_products} products.")
        conn.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Processing Reviews: Sentiment, Keywords, and Summaries")
    print("=" * 60)
    
    processor = ReviewProcessor('reviews.db')
    
    # Process sentiment for all reviews
    processor.process_all_reviews()
    
    # Process products for keywords and summaries
    processor.process_products()
    
    print("\nProcessing complete!")


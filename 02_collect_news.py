import requests
import csv
import json
import logging
import re
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

# Advanced scraping tools
from newspaper import Article
from requests_html import HTMLSession
from fake_useragent import UserAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SolanaNewsCollector:
    def __init__(self):
        """Initialize the incremental news collector"""
        # SERP API setup from environment variables
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.serp_base_url = os.getenv("SERP_BASE_URL", "https://csearch.vela.partners/search")
        
        if not self.serp_api_key:
            raise ValueError("SERP_API_KEY environment variable is required")
        
        self.serp_headers = {
            'accept': 'application/json',
            'serp-vela-key': self.serp_api_key
        }
        
        # OpenAI o3-mini setup from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Web scraping headers
        self.scrape_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def read_training_data(self, csv_file: str = "training_set.csv") -> List[Dict]:
        """Read the current state of the training CSV"""
        rows = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
    
        except FileNotFoundError:
            logger.error(f"❌ CSV file {csv_file} not found!")
        except Exception as e:
            logger.error(f"❌ Error reading CSV: {e}")
        
        return rows
    
    def search_articles_multiple_queries(self, after_date: str, before_date: str) -> List[Dict]:
        """Search for Solana articles using multiple search strategies"""
        
        # Multiple search queries to cast a wider net
        search_queries = [
            f"Solana news after:{after_date} before:{before_date}",
            f"SOL price after:{after_date} before:{before_date}",
            f"Solana blockchain after:{after_date} before:{before_date}",
            f"Solana ecosystem after:{after_date} before:{before_date}",
            f"Solana partnerships after:{after_date} before:{before_date}",
            f"Solana development after:{after_date} before:{before_date}"
        ]
        
        all_articles = []
        seen_urls = set()
        
        for query in search_queries:
            try:
                params = {
                    'query': query,
                    'num_results': 10,
                    'page': 1
                }
                
                response = requests.get(self.serp_base_url, headers=self.serp_headers, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get("results", [])
                
                # Deduplicate by URL
                for article in articles:
                    url = article.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_articles.append(article)
                        
            except Exception as e:
                continue  # Try next query if this one fails
        
        print(f"🔍 Found {len(all_articles)} unique articles across {len(search_queries)} search queries")
        return all_articles
    
    def scrape_article_content(self, url: str) -> Dict:
        """Professional-grade article scraping with multiple fallback strategies"""
        
        # Strategy 1: newspaper3k (best for news articles)
        result = self._scrape_with_newspaper(url)
        if result["success"] and len(result.get("content", "")) > 200:
            return result
            
        # Strategy 2: requests-html with JavaScript rendering
        result = self._scrape_with_requests_html(url)
        if result["success"] and len(result.get("content", "")) > 200:
            return result
            
        # Strategy 3: Basic requests with advanced selectors
        result = self._scrape_with_requests(url)
        if result["success"] and len(result.get("content", "")) > 200:
            return result
        
        # If all strategies failed, return the best attempt
        return {"success": False, "error": "All scraping strategies failed", "content": ""}
    
    def _scrape_with_newspaper(self, url: str) -> Dict:
        """Scrape using newspaper3k (specialized for news)"""
        try:
            ua = UserAgent()
            
            article = Article(url)
            article.set_config({'User-Agent': ua.random})
            article.download()
            article.parse()
            
            # Get HTML for date analysis
            full_html = article.html[:5000] if article.html else ""
            
            return {
                "success": True,
                "content": article.text,
                "content_length": len(article.text),
                "title": article.title or "No title",
                "full_html": full_html,
                "method": "newspaper3k"
            }
            
        except Exception as e:
            return {"success": False, "error": f"newspaper3k failed: {str(e)[:100]}"}
    
    def _scrape_with_requests_html(self, url: str) -> Dict:
        """Scrape using requests-html (handles JavaScript)"""
        try:
            session = HTMLSession()
            ua = UserAgent()
            
            # Professional headers
            headers = {
                'User-Agent': ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            response = session.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            # Render JavaScript
            response.html.render(timeout=15, keep_page=True, scrolldown=1)
            
            # Extract content using comprehensive selectors
            content_selectors = [
                'article', '.article-content', '.post-content', '.entry-content',
                '.content', 'main', '.main-content', '.article-body', '.story-body',
                '.post-body', '.article-text', '.content-body', '.article__content',
                '.story-content', '.article__body', '.entry__content', '.post__content',
                '.content__body', '[data-content]', '[data-article-content]',
                '.markdown-body', '.prose', '.rich-text', '.post-container'
            ]
            
            content_text = ""
            for selector in content_selectors:
                elements = response.html.find(selector)
                if elements:
                    content_text = elements[0].text
                    break
            
            # Fallback to paragraph extraction
            if not content_text or len(content_text) < 100:
                paragraphs = response.html.find('p')
                content_text = ' '.join([p.text for p in paragraphs])
            
            # Clean up text
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            # Get title
            title_element = response.html.find('title')
            title = title_element[0].text if title_element else "No title"
            
            session.close()
            
            return {
                "success": True,
                "content": content_text,
                "content_length": len(content_text),
                "title": title,
                "full_html": response.html.html[:5000],
                "method": "requests-html"
            }
            
        except Exception as e:
            return {"success": False, "error": f"requests-html failed: {str(e)[:100]}"}
    
    def _scrape_with_requests(self, url: str) -> Dict:
        """Fallback scraping with basic requests"""
        try:
            ua = UserAgent()
            
            headers = {
                'User-Agent': ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Referer': 'https://www.google.com/',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", ".sidebar", ".nav"]):
                element.decompose()
            
            # Enhanced content selectors
            content_selectors = [
                'article', '.article-content', '.post-content', '.entry-content',
                '.content', 'main', '.main-content', '.article-body', '.story-body',
                '.post-body', '.article-text', '.content-body', '.article__content',
                '.story-content', '.article__body', '.entry__content', '.post__content',
                '.content__body', '[data-content]', '[data-article-content]',
                '.markdown-body', '.prose', '.rich-text', '.text', '.body-text',
                '.article-wrap', '.entry-summary', '.post-wrap', '[role="main"]'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content_text = content_element.get_text()
                    break
            
            # Fallback strategies
            if not content_text or len(content_text) < 100:
                # Try all paragraphs
                paragraphs = soup.find_all('p')
                if len(paragraphs) >= 3:  # Only if substantial content
                    content_text = ' '.join([p.get_text() for p in paragraphs])
                
                # If still empty, try div content
                if not content_text or len(content_text) < 100:
                    divs = soup.find_all('div', class_=lambda x: x and any(term in x.lower() for term in ['content', 'article', 'post', 'story', 'text']))
                    if divs:
                        content_text = divs[0].get_text()
            
            # Clean up text
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            return {
                "success": True,
                "content": content_text,
                "content_length": len(content_text),
                "title": soup.title.string if soup.title else "No title",
                "full_html": str(soup)[:5000],
                "method": "requests"
            }
            
        except Exception as e:
            return {"success": False, "error": f"requests failed: {str(e)[:100]}"}
    
    def ai_verify_date_in_range(self, article_data: Dict, search_description: str, 
                               target_week_start: str, target_week_end: str, url: str) -> Dict:
        """Use AI to extract and verify publication date from article content and search snippet"""
        html_snippet = article_data.get('full_html', '')[:3000]  # First 3000 chars
        content_snippet = article_data.get('content', '')[:2000]  # First 2000 chars
        title = article_data.get('title', '')
        
        prompt = f"""
You are a date verification expert. Analyze this web article and determine its publication date.

ARTICLE DETAILS:
- URL: {url}
- Title: {title}
- Target Date Range: {target_week_start} to {target_week_end}

SEARCH RESULT DESCRIPTION (often contains publication date):
{search_description}

HTML SNIPPET (contains meta tags, time elements):
{html_snippet}

ARTICLE CONTENT:
{content_snippet}

TASK:
1. Extract the publication date from the SEARCH DESCRIPTION FIRST (this often has the most reliable date info)
2. If not found in snippet, check HTML meta tags, time elements, or article text
3. Determine if this date falls within the range {target_week_start} to {target_week_end}
4. Be very strict about the date range - only articles published within these exact dates should be accepted

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "publication_date": "YYYY-MM-DD or 'not found'",
    "within_range": true or false,
    "confidence": "high" or "medium" or "low",
    "reasoning": "Brief explanation of how you found the date and why it is/isn't in range",
    "date_source": "snippet" or "html" or "content" or "not found"
}}

IMPORTANT: 
- PRIORITIZE the search description for date information - it's often the most accurate
- Only return true for within_range if you're confident the article was published between {target_week_start} and {target_week_end}
- If you can't find a clear publication date anywhere, set within_range to false
- Look for patterns like "3 days ago", "Jan 3, 2024", "2024-01-03", etc.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=400
            )
            response_text = response.choices[0].message.content.strip()
            
            # Robust JSON parsing with multiple fallback strategies
            return self.parse_ai_response(response_text, {
                "publication_date": "not found",
                "within_range": False,
                "confidence": "low",
                "reasoning": "AI parsing failed",
                "date_source": "not found"
            })
            
        except Exception as e:
            return {
                "publication_date": "not found",
                "within_range": False,
                "confidence": "low",
                "reasoning": f"AI error: {str(e)}",
                "date_source": "not found"
            }

    def ai_verify_content_quality(self, article_data: Dict, title: str, url: str) -> Dict:
        """Use AI to verify that the article contains sufficient Solana-related content"""
        content = article_data.get('content', '')
        scraped_title = article_data.get('title', title)
        
        # Skip if content is too short
        if len(content) < 100:
            return {
                "has_sufficient_content": False,
                "reasoning": f"Content too short ({len(content)} characters)",
                "confidence": "high"
            }
        
        prompt = f"""
You are evaluating whether this article contains sufficient meaningful content about Solana cryptocurrency.

ARTICLE DETAILS:
- Title: {scraped_title}
- URL: {url}
- Content Length: {len(content)} characters
- Content Preview: {content[:1000]}...

EVALUATION CRITERIA:
1. Does the article contain substantial information specifically about Solana?
2. Is there meaningful content (not just navigation, headers, or generic text)?
3. Does it discuss Solana's technology, price, partnerships, developments, or ecosystem?
4. Is there enough content to extract useful sentiment or data points?

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "has_sufficient_content": true or false,
    "reasoning": "Brief explanation of why the content is/isn't sufficient",
    "confidence": "high" or "medium" or "low"
}}

REQUIREMENTS:
- Return true only if the article has meaningful Solana-specific content (at least 200+ characters of relevant text)
- Return false if it's mostly navigation, headers, generic crypto content, or lacks Solana focus
- Be strict - we want high-quality articles only
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=400
            )
            response_text = response.choices[0].message.content.strip()
            
            return self.parse_ai_response(response_text, {
                "has_sufficient_content": False,
                "reasoning": "AI parsing failed",
                "confidence": "low"
            })
            
        except Exception as e:
            return {
                "has_sufficient_content": False,
                "reasoning": f"AI evaluation error: {str(e)}",
                "confidence": "low"
            }

    def parse_ai_response(self, response_text: str, fallback_dict: Dict) -> Dict:
        """Robust JSON parsing with multiple fallback strategies"""
        if not response_text or response_text.strip() == "":
            return fallback_dict
        
        # Clean the response
        response_clean = response_text.strip()
        
        # Handle markdown code blocks
        if response_clean.startswith('```json'):
            response_clean = response_clean.replace('```json', '').replace('```', '').strip()
        elif response_clean.startswith('```'):
            response_clean = response_clean.replace('```', '').strip()
        
        # Try to find JSON in the response if it's embedded in text
        if not response_clean.startswith('{'):
            start_idx = response_clean.find('{')
            end_idx = response_clean.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                response_clean = response_clean[start_idx:end_idx]
            else:
                return fallback_dict
        
        try:
            result = json.loads(response_clean)
            return result
        except json.JSONDecodeError:
            return fallback_dict

    def validate_article_with_llm(self, article_data: Dict, search_description: str, 
                                 target_week_start: str, target_week_end: str, url: str = "") -> Dict:
        """Two-step validation: date verification + content quality"""
        if not article_data.get("success"):
            return {"valid": False, "reason": "Scraping failed"}
        
        # Step 1: Date verification
        date_result = self.ai_verify_date_in_range(
            article_data, search_description, target_week_start, target_week_end, url
        )
        
        if not date_result.get('within_range', False):
            return {
                "valid": False,
                "reason": f"Date not in range: {date_result.get('reasoning', 'Unknown')}",
                "date_verification": date_result
            }
        
        # Step 2: Content quality verification
        quality_result = self.ai_verify_content_quality(
            article_data, search_description, url
        )
        
        if not quality_result.get('has_sufficient_content', False):
            return {
                "valid": False,
                "reason": f"Low quality content: {quality_result.get('reasoning', 'Unknown')}",
                "date_verification": date_result,
                "quality_verification": quality_result
            }
        
        # Both checks passed
        return {
            "valid": True,
            "reasoning": f"Date verified ({date_result.get('publication_date')}) and content quality confirmed",
            "date_verification": date_result,
            "quality_verification": quality_result,
            "solana_relevance": "high"  # If it passed quality check, assume high relevance
        }
    
    def enhance_summary_with_llm(self, existing_summary: str, new_articles: List[Dict]) -> str:
        """Use LLM to enhance existing summary with new article information"""
        if not new_articles:
            return existing_summary
        
        # Prepare new articles text
        new_articles_text = ""
        for i, article in enumerate(new_articles, 1):
            title = article.get('title', 'No title')
            content = article.get('content', '')[:800]  # Limit content
            relevance = article.get('validation', {}).get('solana_relevance', 'unknown')
            new_articles_text += f"Article {i} ({relevance} relevance):\n{title}\n{content}\n\n"
        
        prompt = f"""
You are creating a short summary of Solana news for price prediction.

Here are {len(new_articles)} articles about Solana:

{new_articles_text}

Create a summary with these rules:
1. Write 5-7 simple sentences
2. Focus on price-relevant info: partnerships, developments, price changes, market news
3. Include dates and numbers when available  
4. Use simple words, not complex language
5. Write as one paragraph (no line breaks)
6. Maximum 100 words

Write ONLY the summary paragraph, nothing else:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600
            )
            
            enhanced_summary = response.choices[0].message.content.strip()
            return enhanced_summary
            
        except Exception as e:
            return existing_summary or f"Failed to generate summary from {len(new_articles)} articles"
    
    def process_all_weeks(self, csv_file: str = "training_set.csv"):
        """Process all 20 weeks to generate summaries for each"""
        print("🚀 Processing ALL 20 weeks for comprehensive context generation")
        print("=" * 70)
        
        # Read CSV data
        rows = self.read_training_data(csv_file)
        if not rows:
            print("❌ No data to process")
            return
        
        total_weeks = len(rows)
        completed_weeks = 0
        skipped_weeks = 0
        
        for week_idx, week_data in enumerate(rows, 1):
            target_date = week_data['target_date']
            context_start = week_data['context_start']
            existing_summary = week_data.get('summarized_context', '').strip()
            
            print(f"\n📅 Week {week_idx}/{total_weeks}: {context_start} → {target_date}")
            
            # Skip if summary already exists
            if existing_summary:
                print(f"   ✅ Summary already exists ({len(existing_summary)} chars) - skipping")
                skipped_weeks += 1
                continue
            
            print(f"   🔍 Searching for articles in date range...")
            
            # Search for articles using multiple strategies
            articles = self.search_articles_multiple_queries(context_start, target_date)
            if not articles:
                print("   ❌ No articles found - skipping week")
                continue
            
            print(f"   📰 Found {len(articles)} articles to analyze")
            
            # Process articles until we find 5 valid ones (or exhaust the list)
            valid_articles = []
            processed_count = 0
            target_valid = 5
            
            for article in articles:
                if len(valid_articles) >= target_valid:
                    break
                    
                processed_count += 1
                url = article.get('url', '')
                title = article.get('title', '')
                description = article.get('description', '')
                
                print(f"   📄 Article {processed_count}: {title[:50]}...")
                
                # Skip if no URL
                if not url:
                    print("      ⚠️ No URL - skip")
                    continue
                
                # Scrape content
                scraped = self.scrape_article_content(url)
                if not scraped.get("success"):
                    error = scraped.get('error', 'Unknown')[:50]
                    print(f"      ❌ Scraping failed: {error}")
                    continue
                else:
                    method = scraped.get('method', 'unknown')
                    content_length = scraped.get('content_length', 0)
                    print(f"      ✅ {content_length} chars via {method}")
                
                # Validate with LLM
                validation = self.validate_article_with_llm(
                    scraped, description, context_start, target_date, url
                )
                
                if validation.get('valid', False):
                    article_data = {
                        'title': title,
                        'url': url,
                        'content': scraped.get('content', ''),
                        'validation': validation
                    }
                    valid_articles.append(article_data)
                    print(f"      ✅ VALID ({len(valid_articles)}/{target_valid})")
                else:
                    reason = validation.get('reason', 'Failed')[:40]
                    print(f"      ❌ INVALID: {reason}")
            
            print(f"   📊 Result: {len(valid_articles)} valid articles from {processed_count} processed")
            
            # Generate summary if we have valid articles
            if valid_articles:
                print(f"   🔄 Generating summary...")
                try:
                    enhanced_summary = self.enhance_summary_with_llm("", valid_articles)
                    
                    if enhanced_summary and enhanced_summary.strip():
                        # Update this week's data
                        rows[week_idx - 1]['summarized_context'] = enhanced_summary
                        
                        # Write back to CSV immediately (in case of interruption)
                        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                            fieldnames = rows[0].keys()
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)
                        
                        print(f"   ✅ Summary saved ({len(enhanced_summary)} chars)")
                        completed_weeks += 1
                    else:
                        print("   ❌ Summary generation returned empty result")
                        
                except Exception as e:
                    print(f"   ❌ Summary generation failed: {str(e)}")
            else:
                print("   ❌ No valid articles found - no summary generated")
            
            # Brief pause between weeks to avoid overwhelming APIs
            if week_idx < total_weeks:
                print("   ⏳ Brief pause before next week...")
                time.sleep(2)
        
        print(f"\n🎉 BATCH PROCESSING COMPLETE!")
        print(f"📊 Final Results:")
        print(f"   ✅ Completed: {completed_weeks} weeks")
        print(f"   ⏭️ Skipped: {skipped_weeks} weeks (already had summaries)")
        print(f"   📝 Total processed: {completed_weeks + skipped_weeks}/{total_weeks} weeks")
        
        if completed_weeks + skipped_weeks == total_weeks:
            print(f"🎯 ALL WEEKS NOW HAVE CONTEXT SUMMARIES!")
        else:
            remaining = total_weeks - completed_weeks - skipped_weeks
            print(f"⚠️ {remaining} weeks still need summaries")

    def process_single_week_test(self, csv_file: str = "training_set.csv"):
        """Test the incremental news collection on the first week - continue until 5 valid articles"""
        print("🔍 Processing first week (2024-07-10 to 2024-07-17)")
        print("🎯 Target: Find 5 valid articles")
        
        # Read CSV data
        rows = self.read_training_data(csv_file)
        if not rows:
            print("❌ No data to process")
            return
        
        # Get first week for testing
        first_week = rows[0]
        target_date = first_week['target_date']
        context_start = first_week['context_start']
        existing_summary = first_week.get('summarized_context', '').strip()
        
        # Don't regenerate summary if one already exists
        if existing_summary:
            print(f"✅ Summary already exists ({len(existing_summary)} chars) - skipping")
            print(f"📝 Existing summary: {existing_summary}")
            return
        
        # Search for articles using multiple strategies
        articles = self.search_articles_multiple_queries(context_start, target_date)
        if not articles:
            print("❌ No articles found")
            return
        
        # Process articles until we find 5 valid ones
        valid_articles = []
        processed_count = 0
        
        for article in articles:
            if len(valid_articles) >= 5:
                break
                
            processed_count += 1
            url = article.get('url', '')
            title = article.get('title', '')
            description = article.get('description', '')
            
            print(f"\n📄 Article {processed_count}: {title[:60]}...")
            
            # Skip if no URL
            if not url:
                print("   ⚠️ No URL found, skipping")
                continue
            
            # Scrape content
            scraped = self.scrape_article_content(url)
            if not scraped.get("success"):
                error = scraped.get('error', 'Unknown error')
                print(f"   ❌ Scraping failed: {error}")
                continue
            else:
                method = scraped.get('method', 'unknown')
                content_length = scraped.get('content_length', 0)
                print(f"   ✅ Scraped {content_length} chars using {method}")
            
            # Validate with LLM (two-step: date + content quality)
            validation = self.validate_article_with_llm(
                scraped, description, context_start, target_date, url
            )
            
            if validation.get('valid', False):
                article_data = {
                    'title': title,
                    'url': url,
                    'content': scraped.get('content', ''),
                    'validation': validation
                }
                valid_articles.append(article_data)
                reasoning = validation.get('reasoning', '')
                print(f"   ✅ VALID - {reasoning[:80]}...")
                print(f"      📊 Progress: {len(valid_articles)}/5 valid articles found")
            else:
                reason = validation.get('reason', 'Failed validation')
                
                # More specific error reporting with debugging
                if 'Date not in range' in reason:
                    date_info = validation.get('date_verification', {})
                    pub_date = date_info.get('publication_date', 'unknown')
                    date_reasoning = date_info.get('reasoning', 'no reasoning')
                    print(f"   ❌ INVALID - Wrong date ({pub_date})")
                    print(f"      🔍 Date reasoning: {date_reasoning[:80]}...")
                elif 'Low quality content' in reason:
                    quality_info = validation.get('quality_verification', {})
                    quality_reason = quality_info.get('reasoning', 'insufficient content')
                    print(f"   ❌ INVALID - {quality_reason[:80]}...")
                else:
                    print(f"   ❌ INVALID - {reason}")
        
        print(f"\n📊 Final Results: {len(valid_articles)} valid articles out of {processed_count} processed")
        
        # Generate summary only if we have valid articles
        if valid_articles:
            print(f"\n🔄 Generating summary from {len(valid_articles)} valid articles...")
            try:
                enhanced_summary = self.enhance_summary_with_llm("", valid_articles)
                
                if not enhanced_summary or enhanced_summary.strip() == "":
                    print("❌ Summary generation returned empty result")
                    return
                
                # Update CSV
                rows[0]['summarized_context'] = enhanced_summary
                
                # Write back to CSV
                with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = rows[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                    
                print(f"\n✅ Generated summary ({len(enhanced_summary)} chars)")
                print(f"📝 Summary: {enhanced_summary}")
                
                if len(valid_articles) < 5:
                    print(f"⚠️ Warning: Only found {len(valid_articles)} articles (target was 5)")
                    
            except Exception as e:
                print(f"❌ Summary generation failed: {str(e)}")
        else:
            print("❌ No valid articles found - no summary generated")

# Test execution
if __name__ == "__main__":
    collector = SolanaNewsCollector()
    collector.process_all_weeks("training_set.csv") 
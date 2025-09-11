"""Enhanced Multi-Modal Embedding Strategy for Excel Data.

This module implements a comprehensive embedding strategy specifically designed
for Excel data characteristics, supporting numerical data, textual content,
hierarchical structures, and business domain context.
"""

import asyncio
import logging
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import statistics

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content types for specialized embedding."""
    NUMERICAL = "numerical"
    TEXTUAL = "textual"
    HIERARCHICAL = "hierarchical"
    BUSINESS_DOMAIN = "business_domain"
    MIXED = "mixed"


class EmbeddingModel(Enum):
    """Available embedding models."""
    GENERAL = "all-MiniLM-L6-v2"
    FINANCIAL = "paraphrase-albert-small-v2"
    SCIENTIFIC = "all-mpnet-base-v2"
    MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class ContentAnalysis:
    """Analysis of content for embedding strategy selection."""
    content_type: ContentType
    numerical_ratio: float
    business_terms_count: int
    hierarchy_depth: int
    patterns_detected: List[str]
    domain_indicators: List[str]
    confidence: float


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    primary_model: EmbeddingModel = EmbeddingModel.GENERAL
    fallback_model: Optional[EmbeddingModel] = None
    enable_gpu: bool = True
    precision: str = "float32"  # float32, float16
    cache_enabled: bool = True
    parallel_processing: bool = True
    max_sequence_length: int = 512


class NumericalEmbedder:
    """Specialized embedder for numerical Excel data."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.cache = {}
    
    def embed_numerical_sequence(self, values: List[float], metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embeddings for numerical sequences with statistical features."""
        try:
            # Calculate statistical features
            if not values:
                return np.zeros(384)  # Default dimension
            
            # Basic statistics
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            min_val = min(values)
            max_val = max(values)
            
            # Advanced features
            range_val = max_val - min_val
            skewness = self._calculate_skewness(values)
            kurtosis = self._calculate_kurtosis(values)
            
            # Trend analysis
            trend = self._calculate_trend(values)
            seasonality = self._detect_seasonality(values)
            
            # Business indicators
            is_currency = metadata.get("has_currency_pattern", False)
            is_percentage = metadata.get("has_percentage_pattern", False)
            data_quality = metadata.get("data_quality_score", 0.5)
            
            # Create feature vector
            statistical_features = [
                mean_val, std_val, min_val, max_val, range_val,
                skewness, kurtosis, trend, seasonality,
                float(is_currency), float(is_percentage), data_quality
            ]
            
            # Normalize features
            normalized_features = self._normalize_features(statistical_features)
            
            # Extend to target dimension (384) with pattern encoding
            pattern_features = self._encode_patterns(values, metadata)
            
            # Combine features
            full_embedding = np.concatenate([
                normalized_features,
                pattern_features,
                np.zeros(max(0, 384 - len(normalized_features) - len(pattern_features)))
            ])[:384]
            
            return full_embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in numerical embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of numerical data."""
        if len(values) < 3:
            return 0.0
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return 0.0
        
        n = len(values)
        skew = sum(((x - mean_val) / std_val) ** 3 for x in values) / n
        return skew
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of numerical data."""
        if len(values) < 4:
            return 0.0
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return 0.0
        
        n = len(values)
        kurt = sum(((x - mean_val) / std_val) ** 4 for x in values) / n - 3
        return kurt
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Normalize slope to [-1, 1] range
        max_change = max(values) - min(values)
        if max_change == 0:
            return 0.0
        
        normalized_trend = np.tanh(slope / max_change * len(values))
        return normalized_trend
    
    def _detect_seasonality(self, values: List[float]) -> float:
        """Detect seasonality in numerical data."""
        if len(values) < 12:  # Need at least 12 points for seasonality
            return 0.0
        
        try:
            # Simple seasonality detection using autocorrelation
            mean_val = statistics.mean(values)
            centered = [v - mean_val for v in values]
            
            max_lag = min(len(values) // 4, 12)
            autocorrelations = []
            
            for lag in range(1, max_lag + 1):
                if lag >= len(centered):
                    break
                
                corr = sum(centered[i] * centered[i - lag] for i in range(lag, len(centered)))
                autocorrelations.append(abs(corr))
            
            if not autocorrelations:
                return 0.0
            
            # Return normalized seasonality strength
            max_autocorr = max(autocorrelations)
            var_total = sum(x ** 2 for x in centered)
            
            if var_total == 0:
                return 0.0
            
            seasonality_strength = max_autocorr / var_total
            return min(seasonality_strength, 1.0)
            
        except Exception:
            return 0.0
    
    def _normalize_features(self, features: List[float]) -> np.ndarray:
        """Normalize statistical features."""
        features_array = np.array(features)
        
        # Handle NaN and infinite values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Min-max normalization to [-1, 1]
        max_abs = np.max(np.abs(features_array))
        if max_abs > 0:
            normalized = features_array / max_abs
        else:
            normalized = features_array
        
        return normalized
    
    def _encode_patterns(self, values: List[float], metadata: Dict[str, Any]) -> np.ndarray:
        """Encode detected patterns in numerical data."""
        pattern_features = []
        
        # Pattern indicators from metadata
        patterns = metadata.get("patterns_detected", {})
        
        # Currency pattern
        pattern_features.append(float(patterns.get("currency_count", 0) > 0))
        
        # Phone pattern
        pattern_features.append(float(patterns.get("phone_count", 0) > 0))
        
        # Email pattern
        pattern_features.append(float(patterns.get("email_count", 0) > 0))
        
        # URL pattern
        pattern_features.append(float(patterns.get("url_count", 0) > 0))
        
        # Statistical patterns
        is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1)) if len(values) > 1 else False
        is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1)) if len(values) > 1 else False
        has_zeros = 0.0 in values
        has_negatives = any(v < 0 for v in values)
        
        pattern_features.extend([
            float(is_increasing),
            float(is_decreasing), 
            float(has_zeros),
            float(has_negatives)
        ])
        
        # Extend to desired length (e.g., 32 features)
        while len(pattern_features) < 32:
            pattern_features.append(0.0)
        
        return np.array(pattern_features[:32], dtype=np.float32)


class TextualEmbedder:
    """Specialized embedder for textual Excel content."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = SentenceTransformer(config.primary_model.value)
        self.business_terms = self._load_business_terms()
        self.cache = {}
    
    def _load_business_terms(self) -> Dict[str, List[str]]:
        """Load domain-specific business terms."""
        return {
            "financial": [
                "revenue", "profit", "loss", "expense", "cost", "budget", "forecast",
                "roi", "ebitda", "margin", "cash flow", "investment", "dividend",
                "assets", "liabilities", "equity", "balance sheet", "income statement"
            ],
            "sales": [
                "sales", "revenue", "customer", "client", "lead", "conversion",
                "pipeline", "quota", "territory", "commission", "deal", "prospect"
            ],
            "operations": [
                "efficiency", "productivity", "process", "workflow", "quality",
                "performance", "metrics", "kpi", "sla", "utilization", "capacity"
            ],
            "hr": [
                "employee", "staff", "headcount", "salary", "benefits", "training",
                "performance", "review", "promotion", "retention", "turnover"
            ]
        }
    
    async def embed_text_with_context(
        self, 
        text: str, 
        context: Dict[str, Any] = None
    ) -> np.ndarray:
        """Generate context-aware embeddings for textual content."""
        try:
            if context is None:
                context = {}
            
            # Enhance text with business context
            enhanced_text = self._enhance_with_business_context(text, context)
            
            # Generate embedding
            embedding = await asyncio.to_thread(
                self.model.encode,
                enhanced_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Apply domain-specific adjustments
            domain_weight = self._calculate_domain_weight(text, context)
            quality_weight = context.get("data_quality_score", 0.5)
            
            # Weighted embedding
            adjusted_embedding = embedding * (0.7 + 0.3 * domain_weight * quality_weight)
            
            return adjusted_embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in textual embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def _enhance_with_business_context(self, text: str, context: Dict[str, Any]) -> str:
        """Enhance text with business context for better embeddings."""
        enhanced_parts = [text]
        
        # Add file context
        if "file_name" in context:
            file_name = context["file_name"]
            enhanced_parts.append(f"From file: {file_name}")
        
        # Add sheet context
        if "sheet_name" in context:
            sheet_name = context["sheet_name"]
            enhanced_parts.append(f"Sheet: {sheet_name}")
        
        # Add quality context
        if "data_quality" in context:
            quality = context["data_quality"].get("overall_quality", "unknown")
            enhanced_parts.append(f"Data quality: {quality}")
        
        # Add pattern context
        if "patterns_detected" in context:
            patterns = list(context["patterns_detected"].keys())
            if patterns:
                enhanced_parts.append(f"Contains: {', '.join(patterns)}")
        
        return " | ".join(enhanced_parts)
    
    def _calculate_domain_weight(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate domain-specific weight for the text."""
        text_lower = text.lower()
        domain_scores = {}
        
        # Calculate scores for each domain
        for domain, terms in self.business_terms.items():
            score = sum(1 for term in terms if term in text_lower)
            domain_scores[domain] = score / len(terms) if terms else 0
        
        # Return highest domain score
        max_score = max(domain_scores.values()) if domain_scores else 0.5
        return min(max_score, 1.0)


class HierarchicalEmbedder:
    """Embedder for Excel hierarchical structures."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.cache = {}
    
    def embed_hierarchy(
        self, 
        file_data: Dict[str, Any], 
        sheet_data: Dict[str, Any],
        column_data: Dict[str, Any] = None
    ) -> np.ndarray:
        """Generate embeddings for hierarchical Excel structure."""
        try:
            # File-level features
            file_features = self._extract_file_features(file_data)
            
            # Sheet-level features
            sheet_features = self._extract_sheet_features(sheet_data)
            
            # Column-level features (if provided)
            column_features = self._extract_column_features(column_data) if column_data else []
            
            # Combine hierarchical features
            hierarchy_vector = np.concatenate([
                file_features,
                sheet_features,
                column_features,
                np.zeros(max(0, 384 - len(file_features) - len(sheet_features) - len(column_features)))
            ])[:384]
            
            return hierarchy_vector.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in hierarchical embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def _extract_file_features(self, file_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from file-level metadata."""
        features = []
        
        # File size (normalized)
        file_size_mb = file_data.get("file_size_mb", 0)
        features.append(min(file_size_mb / 100.0, 1.0))  # Normalize to 0-1
        
        # Number of sheets (normalized)
        num_sheets = file_data.get("total_sheets", 1)
        features.append(min(num_sheets / 20.0, 1.0))  # Normalize to 0-1
        
        # Total rows and columns (log-normalized)
        total_rows = file_data.get("total_rows", 0)
        total_cols = file_data.get("total_columns", 0)
        
        features.append(np.log10(max(total_rows, 1)) / 6.0)  # Log-normalize (up to 1M rows)
        features.append(np.log10(max(total_cols, 1)) / 3.0)  # Log-normalize (up to 1K cols)
        
        # File type indicators
        extension = file_data.get("extension", "").lower()
        features.extend([
            float(extension == ".xlsx"),
            float(extension == ".xls"),
            float(extension == ".xlsm"),
            float(extension == ".csv")
        ])
        
        # Processing mode indicators
        processing_mode = file_data.get("processing_mode", "unknown")
        features.extend([
            float(processing_mode == "basic"),
            float(processing_mode == "comprehensive"),
            float(processing_mode == "auto")
        ])
        
        # Extend to target length (32 features)
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def _extract_sheet_features(self, sheet_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from sheet-level metadata."""
        features = []
        
        # Sheet dimensions
        num_rows = sheet_data.get("num_rows", 0)
        num_cols = sheet_data.get("num_cols", 0)
        
        features.append(min(np.log10(max(num_rows, 1)) / 5.0, 1.0))
        features.append(min(np.log10(max(num_cols, 1)) / 2.0, 1.0))
        
        # Data quality indicators
        if "data_quality" in sheet_data:
            quality_data = sheet_data["data_quality"]
            features.append(quality_data.get("completeness_score", 0.5))
            features.append(quality_data.get("consistency_score", 0.5))
            features.append(quality_data.get("validity_score", 0.5))
            features.append(quality_data.get("accuracy_score", 0.5))
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])
        
        # Pattern indicators
        if "patterns_detected" in sheet_data:
            patterns = sheet_data["patterns_detected"]
            features.extend([
                float(patterns.get("email_count", 0) > 0),
                float(patterns.get("phone_count", 0) > 0),
                float(patterns.get("url_count", 0) > 0),
                float(patterns.get("currency_count", 0) > 0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Relationship indicators
        if "relationships" in sheet_data:
            relationships = sheet_data["relationships"]
            correlations = relationships.get("correlations", [])
            features.append(min(len(correlations) / 10.0, 1.0))  # Normalized correlation count
        else:
            features.append(0.0)
        
        # Extend to target length (32 features)
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def _extract_column_features(self, column_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from column-level metadata."""
        features = []
        
        # Data type indicators
        data_type = column_data.get("data_type", "unknown")
        features.extend([
            float(data_type == "numeric"),
            float(data_type == "text"),
            float(data_type == "date"),
            float(data_type == "boolean")
        ])
        
        # Statistical features (if numeric)
        if "statistics" in column_data:
            stats = column_data["statistics"]
            features.extend([
                stats.get("mean", 0.0),
                stats.get("std", 0.0),
                stats.get("skewness", 0.0),
                stats.get("kurtosis", 0.0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Uniqueness and completeness
        features.append(column_data.get("uniqueness_ratio", 0.5))
        features.append(column_data.get("completeness_ratio", 0.5))
        
        # Pattern indicators
        features.extend([
            float(column_data.get("has_email_pattern", False)),
            float(column_data.get("has_phone_pattern", False)),
            float(column_data.get("has_currency_pattern", False)),
            float(column_data.get("has_date_pattern", False))
        ])
        
        # Extend to target length (16 features)
        while len(features) < 16:
            features.append(0.0)
        
        return np.array(features[:16], dtype=np.float32)


class EnhancedEmbeddingStrategy:
    """Main coordinator for enhanced multi-modal embedding strategy."""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        
        # Initialize specialized embedders
        self.numerical_embedder = NumericalEmbedder(self.config)
        self.textual_embedder = TextualEmbedder(self.config)
        self.hierarchical_embedder = HierarchicalEmbedder(self.config)
        
        # Caching
        self.embedding_cache = {}
        self.analysis_cache = {}
        
        logger.info("Enhanced embedding strategy initialized")
    
    async def analyze_content(self, content: str, metadata: Dict[str, Any] = None) -> ContentAnalysis:
        """Analyze content to determine optimal embedding strategy."""
        if metadata is None:
            metadata = {}
        
        # Cache key for analysis
        content_hash = hashlib.md5(f"{content}_{str(metadata)}".encode()).hexdigest()
        if content_hash in self.analysis_cache:
            return self.analysis_cache[content_hash]
        
        try:
            # Analyze numerical content
            numerical_ratio = self._calculate_numerical_ratio(content)
            
            # Count business terms
            business_terms_count = self._count_business_terms(content)
            
            # Analyze hierarchy depth
            hierarchy_depth = metadata.get("hierarchy_depth", 0)
            
            # Detect patterns
            patterns_detected = list(metadata.get("patterns_detected", {}).keys())
            
            # Identify domain indicators
            domain_indicators = self._identify_domain_indicators(content, metadata)
            
            # Determine primary content type
            content_type = self._determine_content_type(
                numerical_ratio, business_terms_count, hierarchy_depth, patterns_detected
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(content_type, numerical_ratio, business_terms_count)
            
            analysis = ContentAnalysis(
                content_type=content_type,
                numerical_ratio=numerical_ratio,
                business_terms_count=business_terms_count,
                hierarchy_depth=hierarchy_depth,
                patterns_detected=patterns_detected,
                domain_indicators=domain_indicators,
                confidence=confidence
            )
            
            # Cache analysis
            self.analysis_cache[content_hash] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            return ContentAnalysis(
                content_type=ContentType.MIXED,
                numerical_ratio=0.5,
                business_terms_count=0,
                hierarchy_depth=0,
                patterns_detected=[],
                domain_indicators=[],
                confidence=0.3
            )
    
    async def generate_embedding(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        force_content_type: Optional[ContentType] = None
    ) -> np.ndarray:
        """Generate optimized embedding based on content analysis."""
        if metadata is None:
            metadata = {}
        
        # Cache key
        cache_key = hashlib.md5(
            f"{content}_{str(metadata)}_{str(force_content_type)}".encode()
        ).hexdigest()
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Analyze content if type not forced
            if force_content_type:
                content_type = force_content_type
            else:
                analysis = await self.analyze_content(content, metadata)
                content_type = analysis.content_type
            
            # Generate embedding based on content type
            if content_type == ContentType.NUMERICAL:
                embedding = await self._generate_numerical_embedding(content, metadata)
            elif content_type == ContentType.TEXTUAL:
                embedding = await self._generate_textual_embedding(content, metadata)
            elif content_type == ContentType.HIERARCHICAL:
                embedding = await self._generate_hierarchical_embedding(content, metadata)
            elif content_type == ContentType.BUSINESS_DOMAIN:
                embedding = await self._generate_business_embedding(content, metadata)
            else:  # MIXED
                embedding = await self._generate_mixed_embedding(content, metadata)
            
            # Cache embedding
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback to basic text embedding
            return await self.textual_embedder.embed_text_with_context(content, metadata)
    
    async def _generate_numerical_embedding(self, content: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embedding optimized for numerical content."""
        try:
            # Extract numerical values
            numerical_values = self._extract_numerical_values(content)
            
            if numerical_values:
                return self.numerical_embedder.embed_numerical_sequence(numerical_values, metadata)
            else:
                # Fallback to textual embedding
                return await self.textual_embedder.embed_text_with_context(content, metadata)
        except Exception:
            return await self.textual_embedder.embed_text_with_context(content, metadata)
    
    async def _generate_textual_embedding(self, content: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embedding optimized for textual content."""
        return await self.textual_embedder.embed_text_with_context(content, metadata)
    
    async def _generate_hierarchical_embedding(self, content: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embedding optimized for hierarchical content."""
        try:
            # Combine hierarchical structure embedding with textual content
            hierarchy_embedding = self.hierarchical_embedder.embed_hierarchy(
                metadata.get("file_data", {}),
                metadata.get("sheet_data", {}),
                metadata.get("column_data", {})
            )
            
            # Get textual embedding
            text_embedding = await self.textual_embedder.embed_text_with_context(content, metadata)
            
            # Weighted combination
            combined = 0.6 * text_embedding + 0.4 * hierarchy_embedding
            return combined.astype(np.float32)
            
        except Exception:
            return await self.textual_embedder.embed_text_with_context(content, metadata)
    
    async def _generate_business_embedding(self, content: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embedding optimized for business domain content."""
        # Enhanced business context
        business_metadata = {
            **metadata,
            "is_business_domain": True,
            "domain_weight": 1.0
        }
        return await self.textual_embedder.embed_text_with_context(content, business_metadata)
    
    async def _generate_mixed_embedding(self, content: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for mixed content."""
        try:
            # Generate multiple embeddings
            text_embedding = await self.textual_embedder.embed_text_with_context(content, metadata)
            
            # Try numerical embedding if numbers present
            numerical_values = self._extract_numerical_values(content)
            if numerical_values:
                num_embedding = self.numerical_embedder.embed_numerical_sequence(numerical_values, metadata)
                # Weighted combination
                combined = 0.7 * text_embedding + 0.3 * num_embedding
                return combined.astype(np.float32)
            
            return text_embedding
            
        except Exception:
            return await self.textual_embedder.embed_text_with_context(content, metadata)
    
    def _calculate_numerical_ratio(self, content: str) -> float:
        """Calculate ratio of numerical content."""
        import re
        
        # Find all numbers (including decimals, percentages, currencies)
        number_pattern = r'-?\d+(?:\.\d+)?[%$¬£¥]?'
        numbers = re.findall(number_pattern, content)
        
        # Calculate ratio
        total_tokens = len(content.split())
        if total_tokens == 0:
            return 0.0
        
        numerical_ratio = len(numbers) / total_tokens
        return min(numerical_ratio, 1.0)
    
    def _count_business_terms(self, content: str) -> int:
        """Count business-related terms in content."""
        business_terms = [
            "sales", "revenue", "profit", "cost", "expense", "budget", "forecast",
            "customer", "client", "product", "service", "market", "growth",
            "performance", "efficiency", "quality", "productivity", "roi",
            "investment", "return", "margin", "cash", "flow", "balance"
        ]
        
        content_lower = content.lower()
        count = sum(1 for term in business_terms if term in content_lower)
        return count
    
    def _identify_domain_indicators(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Identify domain-specific indicators."""
        indicators = []
        content_lower = content.lower()
        
        # Financial indicators
        if any(term in content_lower for term in ["revenue", "profit", "cost", "budget", "financial"]):
            indicators.append("financial")
        
        # Sales indicators
        if any(term in content_lower for term in ["sales", "customer", "client", "lead"]):
            indicators.append("sales")
        
        # Operations indicators
        if any(term in content_lower for term in ["process", "efficiency", "quality", "performance"]):
            indicators.append("operations")
        
        # HR indicators
        if any(term in content_lower for term in ["employee", "staff", "salary", "hr"]):
            indicators.append("hr")
        
        return indicators
    
    def _determine_content_type(
        self, 
        numerical_ratio: float, 
        business_terms_count: int, 
        hierarchy_depth: int,
        patterns_detected: List[str]
    ) -> ContentType:
        """Determine primary content type based on analysis."""
        
        # Numerical content threshold
        if numerical_ratio > 0.6:
            return ContentType.NUMERICAL
        
        # Business domain threshold
        if business_terms_count > 3:
            return ContentType.BUSINESS_DOMAIN
        
        # Hierarchical content threshold
        if hierarchy_depth > 2:
            return ContentType.HIERARCHICAL
        
        # Pattern-based detection
        if len(patterns_detected) > 0:
            return ContentType.TEXTUAL
        
        # Default to mixed for balanced content
        if 0.2 < numerical_ratio < 0.6:
            return ContentType.MIXED
        
        # Default to textual
        return ContentType.TEXTUAL
    
    def _calculate_confidence(
        self, 
        content_type: ContentType, 
        numerical_ratio: float, 
        business_terms_count: int
    ) -> float:
        """Calculate confidence in content type classification."""
        base_confidence = 0.5
        
        if content_type == ContentType.NUMERICAL and numerical_ratio > 0.7:
            base_confidence = 0.9
        elif content_type == ContentType.BUSINESS_DOMAIN and business_terms_count > 5:
            base_confidence = 0.85
        elif content_type == ContentType.TEXTUAL:
            base_confidence = 0.75
        elif content_type == ContentType.MIXED:
            base_confidence = 0.6
        
        return min(base_confidence, 1.0)
    
    def _extract_numerical_values(self, content: str) -> List[float]:
        """Extract numerical values from text content."""
        import re
        
        # Pattern for numbers (including decimals, negatives)
        number_pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(number_pattern, content)
        
        try:
            numerical_values = [float(match) for match in matches]
            return numerical_values
        except ValueError:
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        return {
            "embedding_cache_size": len(self.embedding_cache),
            "analysis_cache_size": len(self.analysis_cache),
            "total_memory_mb": (
                len(str(self.embedding_cache)) + len(str(self.analysis_cache))
            ) / (1024 * 1024)
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.analysis_cache.clear()
        logger.info("Enhanced embedding strategy caches cleared")
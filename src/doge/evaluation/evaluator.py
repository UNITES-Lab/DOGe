"""Evaluation utilities for mathematical reasoning tasks.

This module provides functions for extracting and comparing mathematical expressions
from model outputs, specifically designed for AIME problems.
"""

import re
from typing import Dict, List, Optional, Union

import sympy
from loguru import logger
from sympy import Basic, MatrixBase

from .math_utils import parse_sympy_expression, sympy_expr_eq

# Regex patterns for extracting answers
BOXED_ANSWER_PATTERN = r'\boxed{([^{}]*)}'
FINAL_ANSWER_PATTERN = r'(?:final answer|answer)[^:]*:[ \t]*(\S.*?(?=\.|$))'
NUMBER_PATTERN = r'(?:^|\s|is[ \t]+)(-?\d+(?:\.\d+)?)(?:$|\s|\.)'
COMPLEX_EXPRESSION_PATTERN = r'(\d+\s*[\*\/]\s*\(\s*[\d\+\-]+\s*\)[\s\d\+\-\*\/\^]*)'
SIMPLE_EXPRESSION_PATTERN = r'(?:^|\s|value[ \t]+is[ \t]+|expression[ \t]+is[ \t]+)([\d\+\-\*\/\^\(\)]+)(?:$|\s|\.)'


class AnswerExtractor:
    """Extracts mathematical answers from text."""

    @staticmethod
    def extract_from_boxed(text: str) -> Optional[str]:
        """Extract content from \\boxed{...} command.
        
        Args:
            text: The text containing LaTeX with boxed content
            
        Returns:
            The content inside the box if found, None otherwise
        """
        return re.findall(BOXED_ANSWER_PATTERN, text)[-1] if re.findall(BOXED_ANSWER_PATTERN, text) else None

    @staticmethod
    def extract_final_answer(text: str) -> Optional[str]:
        """Extract the final answer based on common formats.
        
        Args:
            text: The text containing a final answer declaration
            
        Returns:
            The extracted answer if found, None otherwise
        """
        # Key phrases to look for
        phrases = [
            r'final answer is:[ \t]*([\d\+\-\*\/\^\(\)]+)',
            r'answer is:[ \t]*([\d\+\-\*\/\^\(\)]+)',
            r'final answer[ \t]*:[ \t]*([\d\+\-\*\/\^\(\)]+)',
            r'answer[ \t]*:[ \t]*([\d\+\-\*\/\^\(\)]+)',
        ]

        # Try each phrase pattern
        for pattern in phrases:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].strip()

        # More general pattern as fallback
        matches = re.findall(FINAL_ANSWER_PATTERN, text, re.IGNORECASE)
        if matches:
            content = matches[-1].strip()

            # If content has a boxed expression, extract from it
            boxed_match = re.findall(BOXED_ANSWER_PATTERN, content)
            if boxed_match:
                return boxed_match[-1].strip()

            return content

        return None

    @staticmethod
    def extract_numeric(text: str) -> Optional[str]:
        """Extract a simple numeric answer.
        
        Args:
            text: The text containing a number
            
        Returns:
            The extracted number if found, None otherwise
        """
        # Look for "value is X" or "is X" pattern first
        value_is_pattern = r'value[ \t]+is[ \t]+(-?\d+(?:\.\d+)?)'
        matches = re.findall(value_is_pattern, text)
        if matches:
            return matches[-1].strip()

        # Then try the general number pattern
        matches = re.findall(NUMBER_PATTERN, text)
        if matches:
            return matches[-1].strip()

        return None

    @staticmethod
    def extract_expression(text: str) -> Optional[str]:
        """Extract a mathematical expression.
        
        Args:
            text: The text containing a mathematical expression
            
        Returns:
            The extracted expression if found, None otherwise
        """
        # First try to match complex expressions like "2*(5-1)/2"
        matches = re.findall(COMPLEX_EXPRESSION_PATTERN, text)
        if matches:
            return matches[-1].strip()

        # Extract expressions from phrases like "the value is X" or "the expression is X"
        extraction_patterns = [
            r'value[ \t]+is[ \t]+([\d\+\-\*\/\^\(\)]+)',
            r'expression[ \t]+is[ \t]+([\d\+\-\*\/\^\(\)]+)',
            r'calculate[ \t]+([\d\+\-\*\/\^\(\)]+)',
        ]

        for pattern in extraction_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()

        # Try the simple expression pattern as a fallback
        matches = re.findall(SIMPLE_EXPRESSION_PATTERN, text)
        if matches:
            return matches[-1].strip()

        return None

    @classmethod
    def extract_answer(cls, text: str, use_last_number: bool = False) -> Optional[str]:
        """Extract an answer using multiple strategies.
        
        Args:
            text: The model output text
            use_last_number: If True and no answer is found with regular methods, 
                            extract the last number from the text
            
        Returns:
            The extracted answer if found, None otherwise
        """
        extractors = [
            cls.extract_from_boxed,
            # cls.extract_final_answer,
            # cls.extract_expression,
            cls.extract_numeric
        ]

        for extractor in extractors:
            answer = extractor(text)
            if answer:
                return answer

        # If no answer found and use_last_number is True, extract the last number
        if use_last_number:
            pattern = r"-?\d*\.?\d+"
            numbers = re.findall(pattern, text.replace(",", ""))
            if numbers:
                return numbers[-1]

        return None


class AnswerComparator:
    """Compares extracted answers with reference answers."""

    @staticmethod
    def parse_expression(expr_str: str, timeout: int = 5) -> Union[Basic, MatrixBase, None]:
        """Parse an expression string into a sympy expression.
        
        Args:
            expr_str: The expression string to parse
            timeout: Maximum time in seconds to spend parsing
            
        Returns:
            The parsed sympy expression if successful, None otherwise
        """
        return parse_sympy_expression(expr_str, timeout)

    @staticmethod
    def expressions_equal(
            expr1: Union[Basic, MatrixBase], expr2: Union[Basic, MatrixBase],
            tolerance: float = 1e-6, precision: int = 6
            ) -> bool:
        """Check if two sympy expressions are equal.
        
        Args:
            expr1: The first sympy expression
            expr2: The second sympy expression
            tolerance: Numerical tolerance for floating point comparisons
            precision: Number of decimal places to compare for sympy_expr_eq
            
        Returns:
            True if expressions are equal, False otherwise
        """
        try:
            # Use sympy_expr_eq for comparison if available
            try:
                if sympy_expr_eq(expr1, expr2, precision=precision):
                    return True
            except (ImportError, ModuleNotFoundError):
                # If sympy_expr_eq is not available (missing latex2sympy2_extended), 
                # skip this step and use traditional methods
                pass

            # Fallback to traditional methods
            # Check if both expressions are numbers
            if all(hasattr(expr, 'is_Number') and expr.is_Number for expr in [expr1, expr2]):
                return abs(float(expr1) - float(expr2)) < tolerance

            # Try symbolic comparison
            diff = sympy.simplify(expr1 - expr2)
            return diff.is_zero
        except Exception as e:
            logger.warning(f"Error comparing expressions: {e}")
            return False

    @classmethod
    def compare_answers(cls, predicted: str, reference: str, is_math_task: bool = False, precision: int = 6) -> bool:
        """Compare a predicted answer with a reference answer.
        
        Args:
            predicted: The predicted answer string
            reference: The reference answer string
            is_math_task: Flag indicating if this is a math task (AIME 24/25)
            precision: Number of decimal places to compare for sympy_expr_eq
            
        Returns:
            True if answers match, False otherwise
        """
        if predicted is None or reference is None:
            return False

        # Simple string match
        if predicted.strip() == reference.strip():
            return True

        # Handle numbers with leading zeros in direct string comparison
        if predicted.strip().isdigit() and reference.strip().isdigit():
            if predicted.strip().lstrip('0') == reference.strip().lstrip('0'):
                return True

        # Parse as expressions
        pred_expr = cls.parse_expression(predicted)
        ref_expr = cls.parse_expression(reference)

        # If both could be parsed, compare them
        if pred_expr is not None and ref_expr is not None:
            return cls.expressions_equal(pred_expr, ref_expr, precision=precision if is_math_task else 6)

        # If parsing failed, compare as strings again with normalized whitespace
        return ' '.join(predicted.split()) == ' '.join(reference.split())


def evaluate_predictions(
        predictions: List[str], references: List[str], use_last_number: bool = True,
        is_math_task: bool = False, precision: int = 6
        ) -> Dict[str, float]:
    """Evaluate a list of predictions against references.
    
    Args:
        predictions: List of predicted answer strings
        references: List of reference answer strings
        use_last_number: If True, fall back to extracting the last number from text
                        when other extraction methods fail
        is_math_task: Flag indicating if this is a math task (AIME 24/25)
        precision: Number of decimal places to compare for sympy_expr_eq
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(predictions) != len(references):
        raise ValueError(f"Mismatched lengths: {len(predictions)} predictions vs {len(references)} references")

    correct = 0
    extracted_count = 0

    for pred, ref in zip(predictions, references):
        # Extract answer from prediction
        extracted_pred = AnswerExtractor.extract_answer(pred, use_last_number=use_last_number)
        extracted_ref = AnswerExtractor.extract_answer(ref, use_last_number=use_last_number)

        if extracted_pred:
            extracted_count += 1
            if AnswerComparator.compare_answers(extracted_pred, extracted_ref, is_math_task=is_math_task, precision=precision):
                correct += 1

    results = {
        "accuracy": correct / len(references) if len(references) > 0 else 0.0,
        "extraction_rate": extracted_count / len(predictions) if len(predictions) > 0 else 0.0,
    }

    return results

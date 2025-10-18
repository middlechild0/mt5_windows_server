#!/usr/bin/env python3
"""
ICT AI Knowledge Validation System
==================================

Comprehensive testing system to validate AI understanding of ICT concepts
before advancing to next training module. Uses scenario-based testing,
concept mapping, and practical application challenges.

Features:
- Multi-level validation (understanding, application, integration)
- Scenario-based testing with real market conditions
- Concept dependency verification  
- Performance benchmarking
- Adaptive difficulty based on AI performance

Created: October 2, 2025
"""

import json
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation complexity levels"""
    BASIC_UNDERSTANDING = "basic_understanding"     # Can define concepts
    PRACTICAL_APPLICATION = "practical_application" # Can apply in scenarios
    INTEGRATION = "integration"                     # Can combine multiple concepts
    EXPERT_JUDGMENT = "expert_judgment"             # Can handle edge cases

class ValidationResult(Enum):
    """Validation outcomes"""
    PASSED = "passed"
    CONDITIONAL_PASS = "conditional_pass" 
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"

@dataclass
class ValidationChallenge:
    """Individual validation challenge"""
    id: str
    concept: str
    level: ValidationLevel
    scenario: str
    correct_response: str
    acceptable_alternatives: List[str]
    scoring_criteria: Dict[str, float]
    difficulty_rating: int  # 1-5

@dataclass
class ValidationSession:
    """Complete validation session results"""
    module_name: str
    challenges_attempted: List[str]
    scores: Dict[str, float]
    overall_score: float
    result: ValidationResult
    recommendations: List[str]
    timestamp: str

class ICTKnowledgeValidator:
    """Main validation system for ICT knowledge"""
    
    def __init__(self):
        self.validation_challenges: Dict[str, List[ValidationChallenge]] = {}
        self.performance_thresholds = {
            ValidationLevel.BASIC_UNDERSTANDING: 0.85,
            ValidationLevel.PRACTICAL_APPLICATION: 0.75,
            ValidationLevel.INTEGRATION: 0.70,
            ValidationLevel.EXPERT_JUDGMENT: 0.65
        }
        self._initialize_validation_challenges()
        
    def _initialize_validation_challenges(self):
        """Initialize validation challenges for each module"""
        
        # FOUNDATION MODULE VALIDATION
        foundation_challenges = [
            ValidationChallenge(
                id="F_VAL_001",
                concept="market_structure",
                level=ValidationLevel.BASIC_UNDERSTANDING,
                scenario="EUR/USD daily chart shows: High at 1.1200, Low at 1.1150, High at 1.1250, Low at 1.1180, High at 1.1280. What is the market structure?",
                correct_response="Uptrend - showing higher highs (1.1200→1.1250→1.1280) and higher lows (1.1150→1.1180)",
                acceptable_alternatives=[
                    "Bullish trend with HH and HL pattern",
                    "Upward trending market structure",
                    "Higher highs and higher lows = uptrend"
                ],
                scoring_criteria={
                    "correctly_identifies_trend_direction": 0.4,
                    "recognizes_HH_HL_pattern": 0.4,
                    "proper_terminology": 0.2
                },
                difficulty_rating=2
            ),
            
            ValidationChallenge(
                id="F_VAL_002", 
                concept="break_of_structure",
                level=ValidationLevel.PRACTICAL_APPLICATION,
                scenario="GBP/USD is in downtrend with swing lows at 1.2750, 1.2700, 1.2650. Price suddenly closes at 1.2780. What happened and what should you do?",
                correct_response="Break of structure occurred - price took out previous swing high. Wait for confirmation before trading new bullish direction.",
                acceptable_alternatives=[
                    "BOS to the upside - need confirmation for long entries",
                    "Structure broken, trend might be changing, wait for confirmation",
                    "Bearish trend invalidated, look for bullish confirmation"
                ],
                scoring_criteria={
                    "identifies_BOS_correctly": 0.4,
                    "recognizes_trend_change_implication": 0.3,
                    "mentions_need_for_confirmation": 0.3
                },
                difficulty_rating=3
            ),
            
            ValidationChallenge(
                id="F_VAL_003",
                concept="change_of_character", 
                level=ValidationLevel.INTEGRATION,
                scenario="USD/JPY uptrend: HH at 145.50, HL at 144.80, but instead of making new HH, price drops to 144.50 (below previous HL). Multiple timeframe analysis required.",
                correct_response="Change of Character (CHOCH) - uptrend failed to make HH and broke below previous HL. Signals potential trend reversal. Check higher timeframes for confirmation.",
                acceptable_alternatives=[
                    "CHOCH detected - trend structure compromised, expect reversal",
                    "Uptrend invalidated by lower low formation - bearish CHOCH",
                    "Market structure shifted from bullish to bearish character"
                ],
                scoring_criteria={
                    "identifies_CHOCH_correctly": 0.3,
                    "explains_failure_to_continue_trend": 0.3, 
                    "mentions_reversal_implications": 0.2,
                    "suggests_timeframe_confirmation": 0.2
                },
                difficulty_rating=4
            )
        ]
        
        # BASIC CONCEPTS MODULE VALIDATION
        basic_challenges = [
            ValidationChallenge(
                id="B_VAL_001",
                concept="order_blocks",
                level=ValidationLevel.BASIC_UNDERSTANDING,
                scenario="On EUR/USD 1H chart, you see 5 consecutive green candles creating 80-pip upward move. The candle before this move was red, closing at 1.1150. What is this red candle?",
                correct_response="Bullish order block - the last opposing (red) candle before strong upward displacement. Institutional orders likely filled here.",
                acceptable_alternatives=[
                    "Bullish OB at 1.1150 - last red candle before displacement",
                    "Order block formed by last down-close before big move up",
                    "Institutional order level where smart money went long"
                ],
                scoring_criteria={
                    "identifies_order_block": 0.4,
                    "recognizes_opposing_candle_concept": 0.3,
                    "understands_displacement_requirement": 0.3
                },
                difficulty_rating=2
            ),
            
            ValidationChallenge(
                id="B_VAL_002",
                concept="fair_value_gaps",
                level=ValidationLevel.PRACTICAL_APPLICATION,
                scenario="GBP/JPY shows: Candle 1 (High: 180.50, Low: 180.20), Candle 2 (High: 181.00, Low: 180.60), Candle 3 (High: 181.30, Low: 181.10). Is there a Fair Value Gap?",
                correct_response="Yes, bullish Fair Value Gap exists. Gap from 180.50 (candle 1 high) to 181.10 (candle 3 low). 60-pip imbalance that price may return to fill.",
                acceptable_alternatives=[
                    "Bullish FVG from 180.50 to 181.10 - 60 pip gap",
                    "FVG present - candle 1 high doesn't connect to candle 3 low",
                    "Price imbalance between 180.50-181.10 creates FVG"
                ],
                scoring_criteria={
                    "correctly_identifies_FVG_presence": 0.3,
                    "calculates_gap_levels_accurately": 0.3,
                    "recognizes_bullish_direction": 0.2,
                    "mentions_potential_fill": 0.2
                },
                difficulty_rating=3
            ),
            
            ValidationChallenge(
                id="B_VAL_003",
                concept="liquidity",
                level=ValidationLevel.INTEGRATION,
                scenario="USD/CAD has equal highs at 1.3650 (touched 3 times) and equal lows at 1.3580 (touched 2 times). Where is liquidity and how would smart money likely target it?",
                correct_response="Buy-stop liquidity above 1.3650 highs (retail stops + breakout orders). Sell-stop liquidity below 1.3580 lows. Smart money likely to sweep one side first, then reverse to target the other side.",
                acceptable_alternatives=[
                    "Liquidity pools at 1.3650 (buy stops) and 1.3580 (sell stops) - expect sweeps",
                    "Equal highs/lows create strong liquidity - smart money will hunt stops",
                    "Institutional targeting: sweep 1.3650 or 1.3580 then reverse"
                ],
                scoring_criteria={
                    "identifies_buy_stop_liquidity": 0.25,
                    "identifies_sell_stop_liquidity": 0.25,
                    "understands_smart_money_targeting": 0.25,
                    "mentions_sweep_and_reverse_concept": 0.25
                },
                difficulty_rating=4
            )
        ]
        
        # INTERMEDIATE MODULE VALIDATION  
        intermediate_challenges = [
            ValidationChallenge(
                id="I_VAL_001",
                concept="order_block_entry",
                level=ValidationLevel.PRACTICAL_APPLICATION,
                scenario="EUR/USD 4H bullish order block at 1.1200-1.1220. Price returns to 1.1210. On 15min chart you see: doji, small red candle, then strong green engulfing candle. Your action?",
                correct_response="Enter long position - price returned to OB with lower timeframe confirmation (engulfing candle). Set stop below OB at 1.1195, target liquidity above recent highs.",
                acceptable_alternatives=[
                    "Take long entry with 15min confirmation - stop below OB",
                    "Enter long on engulfing confirmation at order block",
                    "Valid entry setup - OB + LTF confirmation present"
                ],
                scoring_criteria={
                    "recognizes_valid_entry_setup": 0.3,
                    "identifies_lower_timeframe_confirmation": 0.3,
                    "mentions_stop_placement": 0.2,
                    "considers_target_levels": 0.2
                },
                difficulty_rating=4
            ),
            
            ValidationChallenge(
                id="I_VAL_002",
                concept="fvg_trading",
                level=ValidationLevel.EXPERT_JUDGMENT,
                scenario="AUD/USD bearish FVG at 0.6750-0.6780 created during London session. Price returns during NY session, fills 50% of gap to 0.6765, then shows rejection. Market is in strong downtrend. Your analysis?",
                correct_response="Partial fill acceptable in strong trends - 50% fill shows institutional interest. Rejection from 0.6765 confirms bearish FVG acting as resistance. Enter short with tight stop above 0.6780.",
                acceptable_alternatives=[
                    "Partial FVG fill sufficient - rejection confirms resistance, go short",
                    "Strong trend + 50% fill + rejection = valid short setup",
                    "Don't need full fill in trending markets - enter short on rejection"
                ],
                scoring_criteria={
                    "understands_partial_fill_concept": 0.3,
                    "recognizes_trend_context_importance": 0.2,
                    "identifies_rejection_as_confirmation": 0.2,
                    "appropriate_stop_placement": 0.3
                },
                difficulty_rating=5
            )
        ]
        
        # ADVANCED MODULE VALIDATION
        advanced_challenges = [
            ValidationChallenge(
                id="A_VAL_001",
                concept="order_block_invalidation",
                level=ValidationLevel.EXPERT_JUDGMENT,
                scenario="You're long EUR/GBP from bullish OB at 0.8650. Price action: spike to 0.8680, pullback to 0.8660, then sudden drop with 15min candle closing at 0.8645. Position management?",
                correct_response="Order block invalidated by close below 0.8650. Exit immediately - invalidation often leads to significant moves against position. Accept small loss rather than risk larger drawdown.",
                acceptable_alternatives=[
                    "OB broken - exit long immediately, invalidation = danger",
                    "Close below OB = invalidation, cut losses now",
                    "Bullish OB compromised - exit to prevent larger loss"
                ],
                scoring_criteria={
                    "recognizes_invalidation_correctly": 0.4,
                    "understands_urgency_of_exit": 0.3,
                    "mentions_risk_of_continued_move": 0.3
                },
                difficulty_rating=5
            ),
            
            ValidationChallenge(
                id="A_VAL_002",
                concept="risk_management",
                level=ValidationLevel.INTEGRATION,
                scenario="$50,000 account, 2% risk rule. GBP/USD setup: Entry 1.2500, Stop 1.2450 (50 pips), Target 1.2600 (100 pips). Calculate position size and validate setup.",
                correct_response="Risk amount: $1,000 (2% of $50k). Stop distance: 50 pips. Position size: 2.0 lots ($10/pip × 100 pips = $1,000). Setup valid: 1:2 RR ratio, proper position sizing.",
                acceptable_alternatives=[
                    "$1000 risk ÷ 50 pips = 2.0 lots, good 1:2 RR setup", 
                    "Position size: 2 lots based on $1000 risk and 50 pip stop",
                    "Proper sizing: $1000 risk, 50 pip stop = 2.0 lot position"
                ],
                scoring_criteria={
                    "calculates_risk_amount_correctly": 0.3,
                    "determines_position_size_accurately": 0.3,
                    "recognizes_risk_reward_ratio": 0.2,
                    "validates_setup_quality": 0.2
                },
                difficulty_rating=4
            )
        ]
        
        # Assign challenges to modules
        self.validation_challenges = {
            "module_1_foundation": foundation_challenges,
            "module_2_basic": basic_challenges,
            "module_3_intermediate": intermediate_challenges,
            "module_4_advanced": advanced_challenges
        }
        
        logger.info(f"✅ Initialized validation challenges for {len(self.validation_challenges)} modules")
    
    def validate_module_knowledge(self, module_name: str, ai_responses: Optional[Dict[str, str]] = None) -> ValidationSession:
        """Validate AI knowledge for specific module"""
        
        if module_name not in self.validation_challenges:
            logger.error(f"No validation challenges found for module: {module_name}")
            return None
        
        logger.info(f"🔍 Starting validation for {module_name}")
        
        challenges = self.validation_challenges[module_name]
        session_scores = {}
        attempted_challenges = []
        
        # Execute validation challenges
        for challenge in challenges:
            logger.info(f"\n--- Challenge {challenge.id}: {challenge.concept} ---")
            logger.info(f"Scenario: {challenge.scenario}")
            
            # Simulate AI response (in real implementation, this would be actual AI response)
            if ai_responses and challenge.id in ai_responses:
                ai_response = ai_responses[challenge.id]
            else:
                ai_response = self._simulate_ai_response(challenge)
            
            logger.info(f"AI Response: {ai_response}")
            
            # Score the response
            score = self._score_response(challenge, ai_response)
            session_scores[challenge.id] = score
            attempted_challenges.append(challenge.id)
            
            logger.info(f"Score: {score:.2f}")
        
        # Calculate overall results
        overall_score = sum(session_scores.values()) / len(session_scores) if session_scores else 0
        result = self._determine_validation_result(overall_score, session_scores)
        recommendations = self._generate_recommendations(challenges, session_scores)
        
        # Create session summary
        session = ValidationSession(
            module_name=module_name,
            challenges_attempted=attempted_challenges,
            scores=session_scores,
            overall_score=overall_score,
            result=result,
            recommendations=recommendations,
            timestamp=str(datetime.now())
        )
        
        self._log_validation_results(session)
        return session
    
    def _simulate_ai_response(self, challenge: ValidationChallenge) -> str:
        """Simulate AI response for testing (replace with real AI in production)"""
        
        # Simulate varying AI performance based on difficulty
        performance_factor = max(0.3, 1.0 - (challenge.difficulty_rating - 1) * 0.15)
        
        if random.random() < performance_factor:
            # Good response - return acceptable alternative
            if challenge.acceptable_alternatives:
                return random.choice(challenge.acceptable_alternatives)
            else:
                return challenge.correct_response
        else:
            # Poor response - simulate common mistakes
            poor_responses = {
                "market_structure": "I see price moving up and down",
                "order_blocks": "There's a big candle here",
                "fair_value_gaps": "Price has a gap",
                "liquidity": "There are stops somewhere",
                "break_of_structure": "Price broke something",
                "order_block_entry": "Enter when price comes back",
                "risk_management": "Risk 2% of account"
            }
            return poor_responses.get(challenge.concept, "I'm not sure about this scenario")
    
    def _score_response(self, challenge: ValidationChallenge, ai_response: str) -> float:
        """Score AI response against challenge criteria"""
        
        total_score = 0.0
        ai_lower = ai_response.lower()
        
        # Check against correct response
        if self._responses_match(ai_response, challenge.correct_response):
            return 1.0
        
        # Check against acceptable alternatives
        for alternative in challenge.acceptable_alternatives:
            if self._responses_match(ai_response, alternative):
                return 0.85  # Slightly lower than perfect
        
        # Detailed scoring based on criteria
        for criterion, weight in challenge.scoring_criteria.items():
            if self._check_criterion_met(criterion, ai_lower, challenge):
                total_score += weight
        
        return min(total_score, 1.0)
    
    def _responses_match(self, response1: str, response2: str, threshold: float = 0.7) -> bool:
        """Check if responses match sufficiently"""
        
        # Simple keyword-based matching (in production, use more sophisticated NLP)
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words2:  # Avoid division by zero
            return False
            
        overlap = len(words1.intersection(words2))
        similarity = overlap / len(words2)
        
        return similarity >= threshold
    
    def _check_criterion_met(self, criterion: str, ai_response: str, challenge: ValidationChallenge) -> bool:
        """Check if specific criterion is met in response"""
        
        criterion_keywords = {
            "correctly_identifies_trend_direction": ["uptrend", "downtrend", "bullish", "bearish", "up", "down"],
            "recognizes_HH_HL_pattern": ["higher high", "higher low", "lower high", "lower low", "hh", "hl", "lh", "ll"],
            "identifies_BOS_correctly": ["break of structure", "bos", "broke", "structure", "broken"],
            "identifies_order_block": ["order block", "ob", "institutional", "opposing candle"],
            "correctly_identifies_FVG_presence": ["fair value gap", "fvg", "gap", "imbalance"],
            "calculates_gap_levels_accurately": ["180.50", "181.10", "60", "pip"],
            "recognizes_invalidation_correctly": ["invalid", "broken", "close below", "close above"],
            "calculates_risk_amount_correctly": ["1000", "$1000", "2%", "50000"],
            "determines_position_size_accurately": ["2.0", "2 lot", "lots"]
        }
        
        keywords = criterion_keywords.get(criterion, [])
        return any(keyword in ai_response for keyword in keywords)
    
    def _determine_validation_result(self, overall_score: float, session_scores: Dict[str, float]) -> ValidationResult:
        """Determine overall validation result"""
        
        if overall_score >= 0.85:
            return ValidationResult.PASSED
        elif overall_score >= 0.75:
            # Check if any critical concepts failed badly
            critical_failures = [score for score in session_scores.values() if score < 0.6]
            if len(critical_failures) == 0:
                return ValidationResult.CONDITIONAL_PASS
            else:
                return ValidationResult.NEEDS_REVIEW
        elif overall_score >= 0.65:
            return ValidationResult.NEEDS_REVIEW
        else:
            return ValidationResult.FAILED
    
    def _generate_recommendations(self, challenges: List[ValidationChallenge], scores: Dict[str, float]) -> List[str]:
        """Generate specific recommendations based on performance"""
        
        recommendations = []
        
        # Identify weak areas
        weak_challenges = [(challenge, scores[challenge.id]) for challenge in challenges if scores[challenge.id] < 0.7]
        
        if not weak_challenges:
            recommendations.append("Excellent performance across all concepts")
            recommendations.append("Ready to advance to next module")
            return recommendations
        
        # Specific recommendations for weak areas
        concept_recommendations = {
            "market_structure": "Review trend identification and HH/HL patterns",
            "order_blocks": "Practice identifying opposing candles and displacement",
            "fair_value_gaps": "Work on gap calculation and imbalance recognition", 
            "liquidity": "Study stop placement and institutional targeting",
            "break_of_structure": "Focus on confirmation requirements after BOS",
            "order_block_entry": "Practice multi-timeframe entry confirmation",
            "risk_management": "Master position sizing calculations"
        }
        
        for challenge, score in weak_challenges:
            concept = challenge.concept
            if concept in concept_recommendations:
                recommendations.append(f"{concept.title()}: {concept_recommendations[concept]} (Score: {score:.2f})")
        
        # Overall recommendations
        avg_weak_score = sum(score for _, score in weak_challenges) / len(weak_challenges)
        
        if avg_weak_score < 0.5:
            recommendations.append("Consider returning to previous module for reinforcement")
        elif avg_weak_score < 0.65:
            recommendations.append("Additional practice needed before advancing")
        else:
            recommendations.append("Minor gaps identified - targeted review recommended")
        
        return recommendations
    
    def _log_validation_results(self, session: ValidationSession):
        """Log detailed validation results"""
        
        logger.info(f"\n📊 VALIDATION RESULTS - {session.module_name}")
        logger.info("=" * 50)
        logger.info(f"Overall Score: {session.overall_score:.2f}")
        logger.info(f"Result: {session.result.value.upper()}")
        
        logger.info(f"\nDetailed Scores:")
        for challenge_id, score in session.scores.items():
            logger.info(f"  {challenge_id}: {score:.2f}")
        
        logger.info(f"\nRecommendations:")
        for rec in session.recommendations:
            logger.info(f"  • {rec}")
    
    def create_adaptive_challenge(self, concept: str, difficulty: int, previous_performance: float) -> ValidationChallenge:
        """Create adaptive challenge based on previous performance"""
        
        # Adjust difficulty based on performance
        if previous_performance >= 0.9:
            difficulty = min(5, difficulty + 1)  # Increase difficulty
        elif previous_performance < 0.6:
            difficulty = max(1, difficulty - 1)  # Decrease difficulty
        
        # This would generate dynamic challenges in production
        # For now, return a template challenge
        return ValidationChallenge(
            id=f"ADAPTIVE_{concept}_{difficulty}",
            concept=concept,
            level=ValidationLevel.PRACTICAL_APPLICATION,
            scenario=f"Adaptive scenario for {concept} at difficulty {difficulty}",
            correct_response=f"Expected response for {concept}",
            acceptable_alternatives=[],
            scoring_criteria={"understanding": 1.0},
            difficulty_rating=difficulty
        )

def main():
    """Test the validation system"""
    
    print("=== ICT AI Knowledge Validation System ===")
    print("Testing validation capabilities")
    print()
    
    validator = ICTKnowledgeValidator()
    
    # Test foundation module validation
    print("🧪 Testing Foundation Module Validation...")
    foundation_session = validator.validate_module_knowledge("module_1_foundation")
    
    if foundation_session:
        print(f"✅ Foundation validation completed")
        print(f"📊 Score: {foundation_session.overall_score:.2f}")
        print(f"🎯 Result: {foundation_session.result.value}")
    
    # Test basic concepts validation
    print(f"\n🧪 Testing Basic Concepts Module Validation...")
    basic_session = validator.validate_module_knowledge("module_2_basic")
    
    if basic_session:
        print(f"✅ Basic concepts validation completed") 
        print(f"📊 Score: {basic_session.overall_score:.2f}")
        print(f"🎯 Result: {basic_session.result.value}")

if __name__ == "__main__":
    main()
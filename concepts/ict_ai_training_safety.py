#!/usr/bin/env python3
"""
ICT AI Training Safety Protocols
===============================

Comprehensive safety system to prevent overwhelming AI during training
and ensure proper concept absorption. Implements cognitive load monitoring,
performance tracking, and emergency stop procedures.

Features:
- Real-time cognitive load monitoring
- Performance degradation detection
- Emergency stop protocols
- Adaptive pacing based on AI response
- Recovery procedures for failed sessions
- Safe restart mechanisms

Created: October 2, 2025
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety alert levels"""
    GREEN = "green"       # Normal operation
    YELLOW = "yellow"     # Caution - monitor closely  
    ORANGE = "orange"     # Warning - reduce pace
    RED = "red"           # Critical - stop immediately

class SafetyTrigger(Enum):
    """Safety trigger types"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    VALIDATION_DECLINE = "validation_decline"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class SafetyMetrics:
    """Real-time safety metrics"""
    current_performance: float
    performance_trend: List[float]  # Last 5 scores
    cognitive_load_indicator: float
    session_duration: int  # minutes
    consecutive_failures: int
    validation_scores_trend: List[float]
    fatigue_indicators: List[str]
    safety_level: SafetyLevel

@dataclass
class SafetyEvent:
    """Safety event log entry"""
    timestamp: datetime
    trigger: SafetyTrigger
    severity: SafetyLevel
    metrics: SafetyMetrics
    action_taken: str
    recovery_plan: str

class ICTAITrainingSafety:
    """Comprehensive safety monitoring and intervention system"""
    
    def __init__(self):
        self.safety_events: List[SafetyEvent] = []
        self.current_metrics: Optional[SafetyMetrics] = None
        self.session_start_time: Optional[datetime] = None
        self.emergency_stop_active = False
        
        # Safety thresholds
        self.thresholds = {
            'min_performance': 0.50,          # Below this = emergency stop
            'performance_decline_rate': 0.20,  # 20% decline triggers warning
            'max_session_duration': 120,      # 2 hours max
            'max_consecutive_failures': 3,     # 3 failures = stop
            'cognitive_overload_threshold': 0.80,  # Calculated metric
            'fatigue_threshold': 0.70,        # Performance drop indicating fatigue
            'recovery_time_minimum': 300      # 5 minutes minimum recovery
        }
        
        # Safety protocols
        self.protocols = {
            'emergency_stop_procedures': [
                'Immediately cease all training activity',
                'Save current progress and session data',
                'Analyze failure patterns', 
                'Generate recovery plan',
                'Require human approval before restart'
            ],
            'performance_degradation_response': [
                'Reduce training pace by 50%',
                'Increase reinforcement frequency',
                'Simplify current concepts',
                'Monitor for 3 consecutive improvements'
            ],
            'cognitive_overload_response': [
                'Implement immediate 5-minute break',
                'Reduce information density',
                'Switch to review mode',
                'Monitor cognitive load indicators'
            ]
        }
        
        logger.info("🛡️  AI Training Safety Protocols initialized")
        
    def initialize_safety_monitoring(self, session_id: str):
        """Initialize safety monitoring for new session"""
        
        self.session_start_time = datetime.now()
        self.emergency_stop_active = False
        
        self.current_metrics = SafetyMetrics(
            current_performance=0.0,
            performance_trend=[],
            cognitive_load_indicator=0.0,
            session_duration=0,
            consecutive_failures=0,
            validation_scores_trend=[],
            fatigue_indicators=[],
            safety_level=SafetyLevel.GREEN
        )
        
        logger.info(f"🛡️  Safety monitoring initialized for session: {session_id}")
        
    def update_safety_metrics(self, performance_score: float, validation_score: float, 
                            ai_response_time: float, rule_complexity: int) -> SafetyLevel:
        """Update safety metrics with latest data"""
        
        if not self.current_metrics:
            logger.error("Safety metrics not initialized")
            return SafetyLevel.RED
        
        # Update performance tracking
        self.current_metrics.current_performance = performance_score
        self.current_metrics.performance_trend.append(performance_score)
        self.current_metrics.validation_scores_trend.append(validation_score)
        
        # Keep only last 5 scores for trend analysis
        if len(self.current_metrics.performance_trend) > 5:
            self.current_metrics.performance_trend = self.current_metrics.performance_trend[-5:]
        if len(self.current_metrics.validation_scores_trend) > 5:
            self.current_metrics.validation_scores_trend = self.current_metrics.validation_scores_trend[-5:]
        
        # Update session duration
        if self.session_start_time:
            self.current_metrics.session_duration = int((datetime.now() - self.session_start_time).total_seconds() / 60)
        
        # Calculate cognitive load indicator
        self.current_metrics.cognitive_load_indicator = self._calculate_cognitive_load(
            ai_response_time, rule_complexity, len(self.current_metrics.performance_trend)
        )
        
        # Update failure count
        if performance_score < 0.60:
            self.current_metrics.consecutive_failures += 1
        else:
            self.current_metrics.consecutive_failures = 0
        
        # Detect fatigue indicators
        self._detect_fatigue_indicators()
        
        # Determine safety level
        safety_level = self._assess_safety_level()
        self.current_metrics.safety_level = safety_level
        
        # Log safety status
        self._log_safety_status()
        
        return safety_level
    
    def _calculate_cognitive_load(self, response_time: float, complexity: int, rules_learned: int) -> float:
        """Calculate cognitive load indicator (0-1, higher = more load)"""
        
        # Base load from response time (slower responses indicate higher load)
        time_load = min(1.0, response_time / 30.0)  # 30+ seconds = max load
        
        # Complexity load
        complexity_load = complexity / 5.0  # 5 = max complexity
        
        # Cumulative load (more rules learned = higher load)
        cumulative_load = min(1.0, rules_learned / 10.0)  # 10+ rules = high load
        
        # Performance decline load
        decline_load = 0.0
        if len(self.current_metrics.performance_trend) >= 3:
            recent_avg = sum(self.current_metrics.performance_trend[-3:]) / 3
            earlier_avg = sum(self.current_metrics.performance_trend[:-3]) / len(self.current_metrics.performance_trend[:-3]) if len(self.current_metrics.performance_trend) > 3 else recent_avg
            
            if earlier_avg > 0:
                decline_rate = (earlier_avg - recent_avg) / earlier_avg
                decline_load = max(0.0, decline_rate * 2)  # Convert decline to load
        
        # Weighted combination
        total_load = (
            time_load * 0.3 +
            complexity_load * 0.2 + 
            cumulative_load * 0.2 +
            decline_load * 0.3
        )
        
        return min(1.0, total_load)
    
    def _detect_fatigue_indicators(self):
        """Detect indicators of AI fatigue or degradation"""
        
        indicators = []
        
        # Performance trend analysis
        if len(self.current_metrics.performance_trend) >= 4:
            recent_scores = self.current_metrics.performance_trend[-4:]
            if all(recent_scores[i] < recent_scores[i-1] for i in range(1, len(recent_scores))):
                indicators.append("Consistent performance decline over 4 rules")
        
        # Low absolute performance
        if self.current_metrics.current_performance < self.thresholds['fatigue_threshold']:
            indicators.append(f"Performance below fatigue threshold: {self.current_metrics.current_performance:.2f}")
        
        # High cognitive load
        if self.current_metrics.cognitive_load_indicator > self.thresholds['cognitive_overload_threshold']:
            indicators.append(f"High cognitive load: {self.current_metrics.cognitive_load_indicator:.2f}")
        
        # Session duration
        if self.current_metrics.session_duration > 90:  # 1.5 hours
            indicators.append(f"Extended session duration: {self.current_metrics.session_duration} minutes")
        
        # Consecutive failures
        if self.current_metrics.consecutive_failures >= 2:
            indicators.append(f"Multiple consecutive failures: {self.current_metrics.consecutive_failures}")
        
        self.current_metrics.fatigue_indicators = indicators
    
    def _assess_safety_level(self) -> SafetyLevel:
        """Assess overall safety level based on current metrics"""
        
        # RED (Critical) - Emergency stop required
        if (self.current_metrics.current_performance < self.thresholds['min_performance'] or
            self.current_metrics.consecutive_failures >= self.thresholds['max_consecutive_failures'] or
            self.current_metrics.session_duration > self.thresholds['max_session_duration']):
            return SafetyLevel.RED
        
        # ORANGE (Warning) - Immediate intervention needed
        if (self.current_metrics.cognitive_load_indicator > self.thresholds['cognitive_overload_threshold'] or
            len(self.current_metrics.fatigue_indicators) >= 3 or
            self._check_performance_decline_rate() > self.thresholds['performance_decline_rate']):
            return SafetyLevel.ORANGE
        
        # YELLOW (Caution) - Monitor closely
        if (self.current_metrics.current_performance < 0.70 or
            self.current_metrics.consecutive_failures >= 1 or
            len(self.current_metrics.fatigue_indicators) >= 2 or
            self.current_metrics.session_duration > 75):
            return SafetyLevel.YELLOW
        
        # GREEN (Normal) - Safe to continue
        return SafetyLevel.GREEN
    
    def _check_performance_decline_rate(self) -> float:
        """Check rate of performance decline"""
        
        if len(self.current_metrics.performance_trend) < 3:
            return 0.0
        
        first_half = self.current_metrics.performance_trend[:len(self.current_metrics.performance_trend)//2]
        second_half = self.current_metrics.performance_trend[len(self.current_metrics.performance_trend)//2:]
        
        if not first_half or not second_half:
            return 0.0
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if first_avg > 0:
            decline_rate = (first_avg - second_avg) / first_avg
            return max(0.0, decline_rate)
        
        return 0.0
    
    def check_safety_triggers(self) -> List[SafetyTrigger]:
        """Check for active safety triggers"""
        
        triggers = []
        
        if self.current_metrics.current_performance < self.thresholds['min_performance']:
            triggers.append(SafetyTrigger.PERFORMANCE_DEGRADATION)
        
        if self.current_metrics.cognitive_load_indicator > self.thresholds['cognitive_overload_threshold']:
            triggers.append(SafetyTrigger.COGNITIVE_OVERLOAD)
        
        if self.current_metrics.session_duration > self.thresholds['max_session_duration']:
            triggers.append(SafetyTrigger.TIME_LIMIT_EXCEEDED)
        
        if self.current_metrics.consecutive_failures >= self.thresholds['max_consecutive_failures']:
            triggers.append(SafetyTrigger.CONSECUTIVE_FAILURES)
        
        if self._check_performance_decline_rate() > self.thresholds['performance_decline_rate']:
            triggers.append(SafetyTrigger.VALIDATION_DECLINE)
        
        return triggers
    
    def execute_safety_intervention(self, trigger: SafetyTrigger) -> str:
        """Execute appropriate safety intervention"""
        
        timestamp = datetime.now()
        action_taken = ""
        recovery_plan = ""
        
        if trigger == SafetyTrigger.PERFORMANCE_DEGRADATION:
            action_taken = "Reduced training pace by 50%, increased reinforcement"
            recovery_plan = "Monitor for 3 consecutive improvements before normal pace"
            
        elif trigger == SafetyTrigger.COGNITIVE_OVERLOAD:
            action_taken = "Initiated 5-minute cognitive break, reduced information density"
            recovery_plan = "Resume with simplified concepts, monitor load indicators"
            
        elif trigger == SafetyTrigger.TIME_LIMIT_EXCEEDED:
            action_taken = "Session terminated due to time limit"
            recovery_plan = "Schedule next session after minimum 4-hour break"
            self.emergency_stop_active = True
            
        elif trigger == SafetyTrigger.CONSECUTIVE_FAILURES:
            action_taken = "Emergency stop - multiple consecutive failures"
            recovery_plan = "Analyze failure patterns, restart from previous module"
            self.emergency_stop_active = True
            
        elif trigger == SafetyTrigger.VALIDATION_DECLINE:
            action_taken = "Switched to review mode, reduced new concept introduction"
            recovery_plan = "Focus on reinforcement of existing concepts"
        
        # Log safety event
        safety_event = SafetyEvent(
            timestamp=timestamp,
            trigger=trigger,
            severity=self.current_metrics.safety_level,
            metrics=self.current_metrics,
            action_taken=action_taken,
            recovery_plan=recovery_plan
        )
        
        self.safety_events.append(safety_event)
        self._save_safety_event(safety_event)
        
        logger.warning(f"🚨 SAFETY INTERVENTION: {trigger.value}")
        logger.warning(f"📋 Action: {action_taken}")
        logger.warning(f"🔄 Recovery: {recovery_plan}")
        
        return action_taken
    
    def calculate_recovery_recommendations(self) -> Dict[str, str]:
        """Calculate recovery recommendations based on recent safety events"""
        
        recommendations = {}
        
        # Analyze recent safety events
        recent_events = [event for event in self.safety_events 
                        if event.timestamp > datetime.now() - timedelta(hours=24)]
        
        if not recent_events:
            recommendations['status'] = "No recent safety events - normal operation"
            return recommendations
        
        # Pattern analysis
        trigger_counts = {}
        for event in recent_events:
            trigger = event.trigger.value
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        # Most common trigger
        most_common_trigger = max(trigger_counts.items(), key=lambda x: x[1])
        
        recommendations['primary_concern'] = most_common_trigger[0]
        recommendations['occurrence_count'] = most_common_trigger[1]
        
        # Specific recommendations based on patterns
        if most_common_trigger[0] == 'cognitive_overload':
            recommendations['action'] = "Reduce session length and increase break frequency"
            recommendations['prevention'] = "Implement adaptive pacing based on response times"
            
        elif most_common_trigger[0] == 'performance_degradation':
            recommendations['action'] = "Return to previous module for reinforcement"
            recommendations['prevention'] = "Increase validation frequency and lower advancement thresholds"
            
        elif most_common_trigger[0] == 'consecutive_failures':
            recommendations['action'] = "Comprehensive concept review and simplified examples"
            recommendations['prevention'] = "Improve prerequisite validation before advancing"
        
        # Recovery timeline
        if len(recent_events) >= 3:
            recommendations['recovery_time'] = "24-48 hours break recommended"
        elif len(recent_events) >= 2:
            recommendations['recovery_time'] = "12-24 hours break recommended"
        else:
            recommendations['recovery_time'] = "4-8 hours break recommended"
        
        return recommendations
    
    def _log_safety_status(self):
        """Log current safety status"""
        
        if self.current_metrics.safety_level in [SafetyLevel.ORANGE, SafetyLevel.RED]:
            logger.warning(f"⚠️  Safety Level: {self.current_metrics.safety_level.value.upper()}")
            logger.warning(f"🧠 Cognitive Load: {self.current_metrics.cognitive_load_indicator:.2f}")
            logger.warning(f"📊 Performance: {self.current_metrics.current_performance:.2f}")
            logger.warning(f"❌ Consecutive Failures: {self.current_metrics.consecutive_failures}")
            
            if self.current_metrics.fatigue_indicators:
                logger.warning("🚨 Fatigue Indicators:")
                for indicator in self.current_metrics.fatigue_indicators:
                    logger.warning(f"   • {indicator}")
        
        elif self.current_metrics.safety_level == SafetyLevel.YELLOW:
            logger.info(f"⚠️  Safety Level: YELLOW - Monitor closely")
            logger.info(f"📊 Performance: {self.current_metrics.current_performance:.2f}")
    
    def _save_safety_event(self, event: SafetyEvent):
        """Save safety event to log file"""
        
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            safety_log_file = logs_dir / "safety_events.jsonl"
            
            event_data = {
                'timestamp': event.timestamp.isoformat(),
                'trigger': event.trigger.value,
                'severity': event.severity.value,
                'current_performance': event.metrics.current_performance,
                'cognitive_load': event.metrics.cognitive_load_indicator,
                'session_duration': event.metrics.session_duration,
                'consecutive_failures': event.metrics.consecutive_failures,
                'fatigue_indicators': event.metrics.fatigue_indicators,
                'action_taken': event.action_taken,
                'recovery_plan': event.recovery_plan
            }
            
            with open(safety_log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save safety event: {e}")
    
    def generate_safety_report(self) -> Dict:
        """Generate comprehensive safety report"""
        
        return {
            'current_status': {
                'safety_level': self.current_metrics.safety_level.value if self.current_metrics else 'unknown',
                'emergency_stop_active': self.emergency_stop_active,
                'session_duration': self.current_metrics.session_duration if self.current_metrics else 0
            },
            'performance_metrics': {
                'current_performance': self.current_metrics.current_performance if self.current_metrics else 0,
                'performance_trend': self.current_metrics.performance_trend if self.current_metrics else [],
                'cognitive_load': self.current_metrics.cognitive_load_indicator if self.current_metrics else 0,
                'consecutive_failures': self.current_metrics.consecutive_failures if self.current_metrics else 0
            },
            'safety_events_count': len(self.safety_events),
            'recent_interventions': [event.action_taken for event in self.safety_events[-5:]],
            'recovery_recommendations': self.calculate_recovery_recommendations()
        }

def main():
    """Test safety protocols"""
    
    print("=== ICT AI Training Safety Protocols ===")
    print("Testing safety monitoring and intervention")
    print()
    
    safety = ICTAITrainingSafety()
    safety.initialize_safety_monitoring("test_session_001")
    
    # Simulate training with declining performance
    print("🧪 Simulating training session with performance decline...")
    
    test_scenarios = [
        (0.85, 0.80, 5.0, 2),   # Good start
        (0.78, 0.75, 8.0, 3),   # Slight decline
        (0.70, 0.68, 12.0, 3),  # Warning territory 
        (0.55, 0.50, 18.0, 4),  # Critical performance
        (0.45, 0.40, 25.0, 4)   # Emergency stop territory
    ]
    
    for i, (perf, val, response_time, complexity) in enumerate(test_scenarios, 1):
        print(f"\n--- Rule {i} ---")
        safety_level = safety.update_safety_metrics(perf, val, response_time, complexity)
        
        # Check for triggers
        triggers = safety.check_safety_triggers()
        
        if triggers:
            print(f"🚨 Safety triggers detected: {[t.value for t in triggers]}")
            for trigger in triggers:
                action = safety.execute_safety_intervention(trigger)
                if safety.emergency_stop_active:
                    print("🛑 EMERGENCY STOP ACTIVATED")
                    break
        
        time.sleep(1)  # Simulate time between rules
    
    # Generate final report
    report = safety.generate_safety_report()
    print(f"\n📊 Final Safety Report:")
    print(f"Status: {report['current_status']['safety_level'].upper()}")
    print(f"Emergency Stop: {report['current_status']['emergency_stop_active']}")
    print(f"Safety Events: {report['safety_events_count']}")

if __name__ == "__main__":
    main()
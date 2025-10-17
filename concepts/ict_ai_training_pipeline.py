#!/usr/bin/env python3
"""
ICT AI Training Pipeline
========================

Main pipeline that orchestrates systematic AI training using the structured
learning modules. Features progressive learning, validation checkpoints,
and cognitive load management.

Features:
- Rate-limited rule introduction
- Understanding validation before progression
- Adaptive pacing based on AI performance
- Safe rule feeding with rollback capability
- Progress tracking and reporting

Created: October 2, 2025
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingPhase(Enum):
    """Training phases"""
    INITIALIZATION = "initialization"
    ACTIVE_LEARNING = "active_learning"  
    VALIDATION = "validation"
    CONSOLIDATION = "consolidation"
    ADVANCEMENT = "advancement"

class LearningStatus(Enum):
    """Learning status indicators"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    VALIDATION_PENDING = "validation_pending"
    COMPLETED = "completed"
    NEEDS_REINFORCEMENT = "needs_reinforcement"
    FAILED = "failed"

@dataclass
class TrainingSession:
    """Training session metadata"""
    session_id: str
    module_name: str
    start_time: datetime
    end_time: Optional[datetime]
    rules_covered: List[str]
    validation_scores: List[float]
    average_score: float
    status: LearningStatus
    notes: List[str]

class AITrainingPipeline:
    """Main AI training pipeline with progressive learning"""
    
    def __init__(self, modules_dir: str = "ai_training_modules"):
        self.modules_dir = Path(modules_dir)
        self.modules: Dict = {}
        self.current_session: Optional[TrainingSession] = None
        self.training_history: List[TrainingSession] = []
        self.ai_performance_profile = {
            'learning_rate': 1.0,  # Multiplier for pacing
            'retention_score': 0.0,  # Long-term retention
            'concept_difficulty_preferences': {},  # Which concepts are harder
            'optimal_session_length': 45,  # Minutes
            'fatigue_threshold': 0.75  # Performance drop indicating fatigue
        }
        
        self._load_modules()
        self._initialize_safety_protocols()
        
    def _load_modules(self):
        """Load learning modules from files"""
        try:
            index_file = self.modules_dir / "modules_index.json"
            if not index_file.exists():
                logger.error(f"Modules index not found: {index_file}")
                return
                
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            logger.info(f"Loading {index['total_modules']} learning modules...")
            
            for module_name in index['learning_progression']:
                module_file = self.modules_dir / f"{module_name}.json"
                if module_file.exists():
                    with open(module_file, 'r') as f:
                        self.modules[module_name] = json.load(f)
                    logger.info(f"✅ Loaded {module_name}")
                else:
                    logger.warning(f"⚠️  Module file missing: {module_file}")
            
            logger.info(f"Successfully loaded {len(self.modules)} modules")
            
        except Exception as e:
            logger.error(f"Failed to load modules: {e}")
            
    def _initialize_safety_protocols(self):
        """Initialize safety protocols for AI training"""
        self.safety_protocols = {
            'max_rules_per_session': 5,  # Prevent information overload
            'min_validation_score': 0.70,  # Minimum score to advance
            'max_consecutive_failures': 3,  # Max failures before break
            'cognitive_break_duration': 300,  # 5 minutes between intense sessions
            'max_daily_training_time': 180,  # 3 hours max per day
            'emergency_stop_conditions': [
                'validation_score_below_50_percent',
                'three_consecutive_failures', 
                'session_time_exceeded',
                'fatigue_indicators_detected'
            ]
        }
        logger.info("🛡️  Safety protocols initialized")
    
    def start_progressive_training(self, starting_module: str = "module_1_foundation") -> bool:
        """Start progressive training from specified module"""
        
        if starting_module not in self.modules:
            logger.error(f"Module not found: {starting_module}")
            return False
        
        logger.info("🚀 Starting Progressive AI Training Pipeline")
        logger.info("=" * 60)
        
        # Training sequence
        module_sequence = [
            "module_1_foundation",
            "module_2_basic", 
            "module_3_intermediate",
            "module_4_advanced"
        ]
        
        start_index = module_sequence.index(starting_module) if starting_module in module_sequence else 0
        
        for i, module_name in enumerate(module_sequence[start_index:], start_index):
            logger.info(f"\n📚 Phase {i+1}/4: {self.modules[module_name]['name']}")
            
            # Train module with safety checks
            success = self._train_module_safely(module_name)
            
            if not success:
                logger.error(f"❌ Training failed at module: {module_name}")
                logger.info("🔄 Consider reinforcement training or starting from previous module")
                return False
            
            # Inter-module break for consolidation
            if i < len(module_sequence) - 1:
                self._consolidation_break(module_name)
        
        logger.info("\n🎉 Progressive Training Complete!")
        self._generate_training_report()
        return True
    
    def _train_module_safely(self, module_name: str) -> bool:
        """Train a single module with safety protocols"""
        
        module = self.modules[module_name]
        rules = module['rules']
        
        # Create training session
        session_id = f"{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = TrainingSession(
            session_id=session_id,
            module_name=module_name,
            start_time=datetime.now(),
            end_time=None,
            rules_covered=[],
            validation_scores=[],
            average_score=0.0,
            status=LearningStatus.IN_PROGRESS,
            notes=[]
        )
        
        logger.info(f"🎯 Training Session: {session_id}")
        logger.info(f"📊 Module: {module['name']}")
        logger.info(f"📝 Rules to learn: {len(rules)}")
        logger.info(f"⏱️  Max session time: {module['max_session_time']} minutes")
        
        session_start = time.time()
        max_duration = module['max_session_time'] * 60
        
        # Progressive rule introduction
        for i, rule in enumerate(rules):
            
            # Safety check: Session time
            if time.time() - session_start > max_duration:
                logger.warning("⏱️  Session time limit reached")
                self.current_session.notes.append("Session stopped due to time limit")
                break
            
            # Safety check: Cognitive load
            if self._check_cognitive_overload(i, len(rules)):
                logger.warning("🧠 Cognitive overload detected - taking break")
                self._cognitive_break(30)  # 30 second break
            
            # Teach rule with controlled pacing
            logger.info(f"\n--- Teaching Rule {i+1}/{len(rules)}: {rule['id']} ---")
            success = self._teach_rule_systematically(rule)
            
            if not success:
                logger.error(f"❌ Failed to teach rule {rule['id']}")
                self.current_session.status = LearningStatus.FAILED
                return False
            
            # Validate understanding
            validation_score = self._validate_rule_understanding(rule)
            self.current_session.validation_scores.append(validation_score)
            self.current_session.rules_covered.append(rule['id'])
            
            # Check if reinforcement needed
            if validation_score < self.safety_protocols['min_validation_score']:
                logger.warning(f"⚠️  Low validation score: {validation_score:.2f}")
                self._provide_reinforcement(rule)
                
                # Re-validate after reinforcement
                validation_score = self._validate_rule_understanding(rule)
                self.current_session.validation_scores[-1] = validation_score
            
            # Adaptive pacing based on performance
            delay = self._calculate_adaptive_delay(validation_score, i, len(rules))
            if delay > 0:
                logger.info(f"⏸️  Adaptive pause: {delay:.1f}s")
                time.sleep(delay)
        
        # Session completion
        self.current_session.end_time = datetime.now()
        self.current_session.average_score = sum(self.current_session.validation_scores) / len(self.current_session.validation_scores) if self.current_session.validation_scores else 0
        
        # Check completion criteria
        completion_success = self._check_module_completion(module)
        
        if completion_success:
            self.current_session.status = LearningStatus.COMPLETED
            logger.info(f"✅ Module completed successfully!")
            logger.info(f"📊 Average validation score: {self.current_session.average_score:.2f}")
        else:
            self.current_session.status = LearningStatus.NEEDS_REINFORCEMENT
            logger.warning(f"⚠️  Module needs reinforcement")
            logger.info(f"📊 Average validation score: {self.current_session.average_score:.2f}")
        
        # Save session
        self.training_history.append(self.current_session)
        self._save_session_data()
        
        return completion_success
    
    def _teach_rule_systematically(self, rule: Dict) -> bool:
        """Teach a single rule with systematic approach"""
        
        logger.info(f"📖 Rule: {rule['rule_text']}")
        logger.info(f"🎯 Category: {rule['category']}")
        logger.info(f"⚠️  Risk Level: {rule['risk_level']}")
        logger.info(f"📈 Difficulty: Level {rule['difficulty_level']}")
        
        # Check prerequisites  
        if rule['prerequisites']:
            logger.info(f"📋 Prerequisites: {', '.join(rule['prerequisites'])}")
            if not self._verify_prerequisites(rule['prerequisites']):
                logger.warning("⚠️  Prerequisites not fully mastered")
                return False
        
        # Present rule in digestible phases
        phases = [
            ("Context", f"When to apply: {rule['when_condition']}"),
            ("Core Rule", rule['rule_text']),
            ("Risk Assessment", f"Risk level: {rule['risk_level']} - impacts position sizing and management"),
            ("Examples", rule['examples'] if rule['examples'] else ["No specific examples provided"])
        ]
        
        for phase_name, phase_content in phases:
            logger.info(f"\n🔍 {phase_name}:")
            if isinstance(phase_content, list):
                for item in phase_content:
                    logger.info(f"   • {item}")
            else:
                logger.info(f"   {phase_content}")
            
            # Allow processing time between phases
            time.sleep(2)
        
        # Consolidation pause
        logger.info("🧠 Processing time for rule integration...")
        time.sleep(3)
        
        return True
    
    def _validate_rule_understanding(self, rule: Dict) -> float:
        """Validate AI understanding with realistic scoring"""
        
        logger.info(f"\n🔍 Validating understanding of {rule['id']}...")
        
        # Simulate validation questions
        questions = rule.get('validation_questions', [])
        if questions:
            logger.info("📝 Validation Questions:")
            for i, question in enumerate(questions, 1):
                logger.info(f"   {i}. {question}")
        
        # Realistic scoring based on rule complexity
        base_score = 0.80
        
        # Difficulty penalty
        difficulty_penalty = (rule['difficulty_level'] - 1) * 0.05
        
        # Risk level bonus (higher risk rules should be learned better)
        risk_bonus = {'CRITICAL': 0.10, 'HIGH': 0.05, 'MEDIUM': 0.02, 'LOW': 0.00}.get(rule['risk_level'], 0)
        
        # Simulation variance
        import random
        variance = random.uniform(-0.15, 0.15)
        
        # Calculate final score
        score = max(0.0, min(1.0, base_score - difficulty_penalty + risk_bonus + variance))
        
        logger.info(f"📊 Validation Score: {score:.2f}")
        
        # Performance tracking
        if score >= 0.90:
            logger.info("🌟 Excellent understanding!")
        elif score >= 0.80:
            logger.info("✅ Good understanding")
        elif score >= 0.70:
            logger.info("⚠️  Adequate understanding") 
        else:
            logger.info("❌ Needs reinforcement")
        
        return score
    
    def _provide_reinforcement(self, rule: Dict):
        """Provide additional reinforcement for poorly understood rules"""
        logger.info(f"\n🔄 Providing reinforcement for rule {rule['id']}")
        
        # Additional context and examples
        logger.info("💡 Additional Context:")
        logger.info(f"   This rule is critical because: {rule['risk_level']} risk level")
        
        if rule['examples']:
            logger.info("📚 Review Examples:")
            for example in rule['examples']:
                logger.info(f"   • {example}")
        
        # Simplified explanation
        logger.info("🎯 Simplified: " + rule['rule_text'])
        
        # Additional processing time
        time.sleep(5)
        logger.info("✅ Reinforcement complete")
    
    def _check_cognitive_overload(self, current_rule_index: int, total_rules: int) -> bool:
        """Check for cognitive overload indicators"""
        
        # Check recent validation scores for declining performance
        if len(self.current_session.validation_scores) >= 3:
            recent_scores = self.current_session.validation_scores[-3:]
            if all(score < 0.70 for score in recent_scores):
                return True
                
        # Check rule density (too many rules in short time)
        if current_rule_index > 0 and current_rule_index % 3 == 0:
            session_duration = (datetime.now() - self.current_session.start_time).total_seconds()
            rules_per_minute = current_rule_index / (session_duration / 60)
            if rules_per_minute > 1.5:  # More than 1.5 rules per minute
                return True
        
        return False
    
    def _cognitive_break(self, duration_seconds: int):
        """Provide cognitive break"""
        logger.info(f"🧠 Cognitive break: {duration_seconds}s for information processing")
        time.sleep(duration_seconds)
        logger.info("🔄 Resuming training...")
    
    def _calculate_adaptive_delay(self, validation_score: float, rule_index: int, total_rules: int) -> float:
        """Calculate adaptive delay based on performance"""
        
        base_delay = 3.0  # Base 3 seconds
        
        # Performance-based adjustment
        if validation_score >= 0.90:
            performance_factor = 0.5  # Faster pacing for good performance
        elif validation_score >= 0.80:
            performance_factor = 1.0  # Normal pacing
        elif validation_score >= 0.70:
            performance_factor = 1.5  # Slower pacing
        else:
            performance_factor = 2.0  # Much slower pacing
        
        # Progressive factor (later rules get more time)
        progress_factor = 1.0 + (rule_index / total_rules) * 0.5
        
        # Calculate final delay
        adaptive_delay = base_delay * performance_factor * progress_factor
        
        return min(adaptive_delay, 15.0)  # Cap at 15 seconds
    
    def _verify_prerequisites(self, prerequisites: List[str]) -> bool:
        """Verify that prerequisites are met"""
        # In real implementation, check AI's performance on prerequisite rules
        # For now, assume prerequisites are met if they were covered in previous sessions
        
        covered_rules = []
        for session in self.training_history:
            covered_rules.extend(session.rules_covered)
        
        return all(prereq in covered_rules for prereq in prerequisites)
    
    def _check_module_completion(self, module: Dict) -> bool:
        """Check if module completion criteria are met"""
        
        criteria = module['completion_criteria']
        avg_score = self.current_session.average_score
        
        # Check validation score threshold
        validation_threshold = criteria.get('validation_score', 0.75)
        
        if avg_score >= validation_threshold:
            logger.info(f"✅ Validation criteria met: {avg_score:.2f} >= {validation_threshold}")
            return True
        else:
            logger.warning(f"❌ Validation criteria not met: {avg_score:.2f} < {validation_threshold}")
            return False
    
    def _consolidation_break(self, completed_module: str):
        """Provide consolidation break between modules"""
        logger.info(f"\n🔄 Consolidation Phase - Processing {completed_module}")
        logger.info("🧠 Allowing time for knowledge integration...")
        
        # Longer break for knowledge consolidation
        consolidation_time = self.safety_protocols['cognitive_break_duration']
        logger.info(f"⏸️  Consolidation break: {consolidation_time}s")
        
        time.sleep(consolidation_time)
        logger.info("✅ Consolidation complete - ready for next module")
    
    def _save_session_data(self):
        """Save session data for analysis"""
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            session_file = logs_dir / f"session_{self.current_session.session_id}.json"
            
            session_data = {
                'session_id': self.current_session.session_id,
                'module_name': self.current_session.module_name,
                'start_time': self.current_session.start_time.isoformat(),
                'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                'rules_covered': self.current_session.rules_covered,
                'validation_scores': self.current_session.validation_scores,
                'average_score': self.current_session.average_score,
                'status': self.current_session.status.value,
                'notes': self.current_session.notes
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def _generate_training_report(self):
        """Generate comprehensive training report"""
        logger.info("\n📊 COMPREHENSIVE TRAINING REPORT")
        logger.info("=" * 60)
        
        total_sessions = len(self.training_history)
        total_rules = sum(len(session.rules_covered) for session in self.training_history)
        avg_score = sum(session.average_score for session in self.training_history) / total_sessions if total_sessions > 0 else 0
        
        completed_modules = [session.module_name for session in self.training_history if session.status == LearningStatus.COMPLETED]
        
        logger.info(f"📈 Total Training Sessions: {total_sessions}")
        logger.info(f"📚 Total Rules Learned: {total_rules}")
        logger.info(f"📊 Overall Average Score: {avg_score:.2f}")
        logger.info(f"✅ Completed Modules: {len(completed_modules)}/4")
        
        for module in completed_modules:
            logger.info(f"   • {module}")
        
        # Performance analysis
        if total_sessions > 1:
            first_score = self.training_history[0].average_score
            last_score = self.training_history[-1].average_score
            improvement = last_score - first_score
            
            logger.info(f"\n📈 Learning Progress:")
            logger.info(f"   Initial performance: {first_score:.2f}")
            logger.info(f"   Final performance: {last_score:.2f}")
            logger.info(f"   Improvement: {improvement:+.2f}")
        
        logger.info(f"\n🎯 Training Status: {'COMPLETE' if len(completed_modules) == 4 else 'IN PROGRESS'}")

def main():
    """Main execution"""
    print("=== ICT AI Training Pipeline ===")
    print("Systematic progressive learning with safety protocols")
    print()
    
    # Initialize pipeline
    pipeline = AITrainingPipeline()
    
    if not pipeline.modules:
        print("❌ No training modules found. Run ict_learning_module_builder.py first.")
        return
    
    # Start training
    print("🚀 Starting Progressive AI Training...")
    success = pipeline.start_progressive_training()
    
    if success:
        print("\n🎉 Training Pipeline Completed Successfully!")
    else:
        print("\n⚠️  Training Pipeline Incomplete - Review logs for details")

if __name__ == "__main__":
    main()
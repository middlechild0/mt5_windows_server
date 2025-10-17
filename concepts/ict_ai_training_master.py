#!/usr/bin/env python3
"""
ICT AI Systematic Training Master
=================================

Master orchestrator that coordinates all components of the systematic
AI training system. Provides a unified interface for progressive
ICT methodology learning with comprehensive safety protocols.

Components Orchestrated:
- Learning Module Builder (converts safe rules to modules)
- Training Pipeline (progressive rule introduction)
- Knowledge Validator (understanding verification)
- Safety Protocols (cognitive load & performance monitoring)

Usage:
    python ict_ai_training_master.py --mode setup      # Build modules
    python ict_ai_training_master.py --mode train      # Start training
    python ict_ai_training_master.py --mode validate   # Validate knowledge
    python ict_ai_training_master.py --mode report     # Generate report

Created: October 2, 2025
"""

import sys
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

# Import our training components
from ict_learning_module_builder import LearningModuleBuilder
from ict_ai_training_pipeline import AITrainingPipeline
from ict_ai_knowledge_validator import ICTKnowledgeValidator
from ict_ai_training_safety import ICTAITrainingSafety

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICTAITrainingMaster:
    """Master coordinator for systematic ICT AI training"""
    
    def __init__(self):
        self.module_builder = None
        self.training_pipeline = None
        self.knowledge_validator = None
        self.safety_system = None
        
        self.system_status = {
            'modules_built': False,
            'training_ready': False,
            'safety_initialized': False,
            'last_training_session': None,
            'total_training_time': 0,
            'modules_completed': 0
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all training components"""
        
        logger.info("🚀 Initializing ICT AI Training Master System")
        logger.info("=" * 60)
        
        try:
            # Initialize components
            self.knowledge_validator = ICTKnowledgeValidator()
            self.safety_system = ICTAITrainingSafety()
            
            # Check if modules exist
            modules_dir = Path("ai_training_modules")
            if modules_dir.exists() and (modules_dir / "modules_index.json").exists():
                self.training_pipeline = AITrainingPipeline()
                self.system_status['modules_built'] = True
                self.system_status['training_ready'] = True
                logger.info("✅ Existing training modules found - system ready")
            else:
                logger.info("📚 No training modules found - run setup mode first")
            
            self.system_status['safety_initialized'] = True
            logger.info("✅ System initialization complete")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def setup_training_system(self):
        """Setup the training system by building modules from safe rules"""
        
        logger.info("\n🔧 SETUP MODE: Building Training System")
        logger.info("=" * 50)
        
        # Initialize module builder
        self.module_builder = LearningModuleBuilder()
        
        # Build learning modules from safe rules
        logger.info("📚 Building learning modules from safe ICT rules...")
        modules = self.module_builder.build_learning_modules()
        
        if not modules:
            logger.error("❌ Failed to build learning modules")
            return False
        
        # Save modules to files
        self.module_builder.save_modules_to_files()
        
        # Initialize training pipeline with new modules
        self.training_pipeline = AITrainingPipeline()
        
        # Update system status
        self.system_status['modules_built'] = True
        self.system_status['training_ready'] = True
        
        logger.info("✅ Training system setup complete!")
        self._print_system_summary()
        
        return True
    
    def start_progressive_training(self, starting_module: str = "module_1_foundation"):
        """Start the systematic progressive training"""
        
        logger.info("\n🎓 TRAINING MODE: Starting Progressive Learning")
        logger.info("=" * 55)
        
        if not self.system_status['training_ready']:
            logger.error("❌ Training system not ready. Run setup mode first.")
            return False
        
        # Initialize safety monitoring
        session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.safety_system.initialize_safety_monitoring(session_id)
        
        # Start progressive training with safety integration
        logger.info("🛡️  Safety protocols active")
        logger.info("📈 Starting progressive training pipeline...")
        
        success = self._train_with_safety_monitoring(starting_module)
        
        # Update system status
        self.system_status['last_training_session'] = datetime.now().isoformat()
        
        if success:
            logger.info("\n🎉 Progressive training completed successfully!")
            self._generate_completion_report()
        else:
            logger.warning("\n⚠️  Training session incomplete - check safety logs")
            self._generate_incident_report()
        
        return success
    
    def _train_with_safety_monitoring(self, starting_module: str) -> bool:
        """Execute training with integrated safety monitoring"""
        
        # Get module sequence
        module_sequence = [
            "module_1_foundation",
            "module_2_basic", 
            "module_3_intermediate",
            "module_4_advanced"
        ]
        
        start_index = module_sequence.index(starting_module) if starting_module in module_sequence else 0
        
        for i, module_name in enumerate(module_sequence[start_index:], start_index):
            
            logger.info(f"\n📚 Module {i+1}/4: Training {module_name}")
            
            # Pre-module validation (except for first module)
            if i > 0:
                previous_module = module_sequence[i-1]
                if not self._validate_prerequisites(previous_module):
                    logger.error(f"❌ Prerequisites not met for {module_name}")
                    return False
            
            # Train module with safety monitoring
            module_success = self._train_single_module_safely(module_name)
            
            if not module_success:
                logger.error(f"❌ Module training failed: {module_name}")
                return False
            
            # Post-module validation
            validation_success = self._validate_module_completion(module_name)
            
            if not validation_success:
                logger.warning(f"⚠️  Module validation failed: {module_name}")
                return False
            
            self.system_status['modules_completed'] += 1
            
            # Inter-module safety break
            if i < len(module_sequence) - 1:
                logger.info("🔄 Inter-module consolidation break...")
                import time
                time.sleep(10)  # Consolidation pause
        
        return True
    
    def _train_single_module_safely(self, module_name: str) -> bool:
        """Train single module with safety monitoring"""
        
        try:
            # Get module data
            module = self.training_pipeline.modules[module_name]
            rules = module['rules']
            
            logger.info(f"🎯 Training {len(rules)} rules in {module_name}")
            
            session_start = datetime.now()
            
            for i, rule in enumerate(rules):
                
                # Safety check before each rule
                if self.safety_system.emergency_stop_active:
                    logger.error("🛑 Emergency stop active - halting training")
                    return False
                
                # Simulate training rule (in real implementation, this would be actual AI training)
                logger.info(f"   📖 Rule {i+1}: {rule['id']} - {rule['category']}")
                
                # Simulate performance metrics
                import random
                performance_score = max(0.3, random.uniform(0.6, 0.95) - (i * 0.02))  # Gradual decline simulation
                validation_score = performance_score + random.uniform(-0.1, 0.1)
                response_time = random.uniform(3, 15) + (i * 0.5)  # Increasing response time
                
                # Update safety metrics
                safety_level = self.safety_system.update_safety_metrics(
                    performance_score, validation_score, response_time, rule['difficulty_level']
                )
                
                # Check for safety interventions
                triggers = self.safety_system.check_safety_triggers()
                
                if triggers:
                    logger.warning(f"🚨 Safety triggers detected for rule {rule['id']}")
                    
                    for trigger in triggers:
                        action = self.safety_system.execute_safety_intervention(trigger)
                        
                        if self.safety_system.emergency_stop_active:
                            logger.error("🛑 Emergency stop triggered during training")
                            return False
                        
                        # Apply intervention (adjust pacing, provide breaks, etc.)
                        self._apply_safety_intervention(action)
                
                # Progressive delay based on safety level
                delay = self._calculate_safety_delay(safety_level)
                if delay > 0:
                    import time
                    time.sleep(delay)
            
            # Calculate session duration
            session_duration = (datetime.now() - session_start).total_seconds() / 60
            self.system_status['total_training_time'] += session_duration
            
            logger.info(f"✅ Module {module_name} training completed ({session_duration:.1f} minutes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during module training: {e}")
            return False
    
    def _apply_safety_intervention(self, action: str):
        """Apply safety intervention action"""
        
        if "break" in action.lower():
            logger.info("⏸️  Applying safety break...")
            import time
            time.sleep(5)  # Safety break
        
        elif "reduce pace" in action.lower():
            logger.info("🐌 Reducing training pace...")
            import time
            time.sleep(3)  # Slower pacing
        
        elif "review mode" in action.lower():
            logger.info("🔄 Switching to review mode...")
            # In real implementation, would switch to reinforcement
    
    def _calculate_safety_delay(self, safety_level) -> float:
        """Calculate delay based on safety level"""
        
        from ict_ai_training_safety import SafetyLevel
        
        delays = {
            SafetyLevel.GREEN: 1.0,   # Normal pace
            SafetyLevel.YELLOW: 2.0,  # Cautious pace
            SafetyLevel.ORANGE: 4.0,  # Slow pace
            SafetyLevel.RED: 0.0      # Stop (handled by emergency stop)
        }
        
        return delays.get(safety_level, 1.0)
    
    def _validate_prerequisites(self, previous_module: str) -> bool:
        """Validate prerequisites are met before advancing"""
        
        logger.info(f"🔍 Validating prerequisites from {previous_module}")
        
        # Use knowledge validator to check understanding
        validation_session = self.knowledge_validator.validate_module_knowledge(previous_module)
        
        if not validation_session:
            return False
        
        # Check if validation passed
        from ict_ai_knowledge_validator import ValidationResult
        
        if validation_session.result in [ValidationResult.PASSED, ValidationResult.CONDITIONAL_PASS]:
            logger.info(f"✅ Prerequisites met (Score: {validation_session.overall_score:.2f})")
            return True
        else:
            logger.warning(f"❌ Prerequisites not met (Score: {validation_session.overall_score:.2f})")
            return False
    
    def _validate_module_completion(self, module_name: str) -> bool:
        """Validate module completion"""
        
        logger.info(f"🎯 Validating completion of {module_name}")
        
        validation_session = self.knowledge_validator.validate_module_knowledge(module_name)
        
        if not validation_session:
            return False
        
        from ict_ai_knowledge_validator import ValidationResult
        
        success = validation_session.result != ValidationResult.FAILED
        
        if success:
            logger.info(f"✅ Module validation passed (Score: {validation_session.overall_score:.2f})")
        else:
            logger.warning(f"❌ Module validation failed (Score: {validation_session.overall_score:.2f})")
            
            # Log recommendations
            for rec in validation_session.recommendations:
                logger.info(f"   💡 {rec}")
        
        return success
    
    def validate_knowledge(self, module_name: str = None):
        """Validate AI knowledge for specific module or all modules"""
        
        logger.info("\n🧪 VALIDATION MODE: Testing Knowledge")
        logger.info("=" * 45)
        
        if not self.system_status['training_ready']:
            logger.error("❌ System not ready for validation")
            return False
        
        modules_to_validate = []
        
        if module_name:
            if module_name in self.knowledge_validator.validation_challenges:
                modules_to_validate = [module_name]
            else:
                logger.error(f"❌ Unknown module: {module_name}")
                return False
        else:
            modules_to_validate = list(self.knowledge_validator.validation_challenges.keys())
        
        # Run validation for each module
        validation_results = {}
        
        for module in modules_to_validate:
            logger.info(f"\n📝 Validating {module}...")
            
            session = self.knowledge_validator.validate_module_knowledge(module)
            
            if session:
                validation_results[module] = {
                    'score': session.overall_score,
                    'result': session.result.value,
                    'recommendations': session.recommendations
                }
        
        # Generate validation report
        self._generate_validation_report(validation_results)
        
        return True
    
    def generate_comprehensive_report(self):
        """Generate comprehensive system report"""
        
        logger.info("\n📊 REPORT MODE: Generating Comprehensive Report")
        logger.info("=" * 55)
        
        report = {
            'system_status': self.system_status,
            'safety_report': self.safety_system.generate_safety_report() if self.safety_system else {},
            'timestamp': datetime.now().isoformat(),
            'training_modules_available': len(self.training_pipeline.modules) if self.training_pipeline else 0
        }
        
        # Save report
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Report saved: {report_file}")
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _print_system_summary(self):
        """Print system setup summary"""
        
        logger.info(f"\n📋 SYSTEM SUMMARY")
        logger.info(f"Modules Built: ✅" if self.system_status['modules_built'] else "❌")
        logger.info(f"Training Ready: ✅" if self.system_status['training_ready'] else "❌")
        logger.info(f"Safety Initialized: ✅" if self.system_status['safety_initialized'] else "❌")
        
        if self.training_pipeline:
            logger.info(f"Available Modules: {len(self.training_pipeline.modules)}")
    
    def _print_report_summary(self, report: dict):
        """Print report summary"""
        
        logger.info(f"\n📈 TRAINING SUMMARY")
        logger.info(f"Modules Completed: {report['system_status']['modules_completed']}/4")
        logger.info(f"Total Training Time: {report['system_status']['total_training_time']:.1f} minutes")
        
        if report['safety_report']:
            safety_status = report['safety_report']['current_status']['safety_level']
            logger.info(f"Safety Status: {safety_status.upper()}")
            logger.info(f"Safety Events: {report['safety_report']['safety_events_count']}")
    
    def _generate_completion_report(self):
        """Generate report for successful completion"""
        
        logger.info(f"\n🎉 TRAINING COMPLETION REPORT")
        logger.info(f"✅ All modules completed successfully")
        logger.info(f"⏱️  Total training time: {self.system_status['total_training_time']:.1f} minutes")
        logger.info(f"🛡️  Safety protocols maintained throughout training")
        logger.info(f"📚 Ready for advanced ICT application training")
    
    def _generate_incident_report(self):
        """Generate report for incomplete training"""
        
        logger.info(f"\n⚠️  TRAINING INCIDENT REPORT")
        logger.info(f"📊 Modules completed: {self.system_status['modules_completed']}/4")
        logger.info(f"⏱️  Training time: {self.system_status['total_training_time']:.1f} minutes")
        
        if self.safety_system:
            recommendations = self.safety_system.calculate_recovery_recommendations()
            logger.info(f"🔄 Recovery recommendations:")
            for key, value in recommendations.items():
                logger.info(f"   {key}: {value}")
    
    def _generate_validation_report(self, results: dict):
        """Generate validation report"""
        
        logger.info(f"\n📝 VALIDATION REPORT")
        
        for module, data in results.items():
            logger.info(f"\n{module}:")
            logger.info(f"  Score: {data['score']:.2f}")
            logger.info(f"  Result: {data['result'].upper()}")
            
            if data['recommendations']:
                logger.info(f"  Recommendations:")
                for rec in data['recommendations'][:3]:  # Top 3
                    logger.info(f"    • {rec}")

def main():
    """Main execution with argument parsing"""
    
    parser = argparse.ArgumentParser(description="ICT AI Systematic Training Master")
    parser.add_argument('--mode', required=True, 
                       choices=['setup', 'train', 'validate', 'report'],
                       help='Operation mode')
    parser.add_argument('--module', 
                       help='Specific module for validation (optional)')
    parser.add_argument('--start-from',
                       default='module_1_foundation',
                       help='Starting module for training')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 ICT AI SYSTEMATIC TRAINING MASTER")
    print("Progressive Learning • Safe Training • Comprehensive Validation")
    print("=" * 70)
    
    try:
        master = ICTAITrainingMaster()
        
        if args.mode == 'setup':
            success = master.setup_training_system()
            if not success:
                sys.exit(1)
                
        elif args.mode == 'train':
            success = master.start_progressive_training(args.start_from)
            if not success:
                sys.exit(1)
                
        elif args.mode == 'validate':
            success = master.validate_knowledge(args.module)
            if not success:
                sys.exit(1)
                
        elif args.mode == 'report':
            master.generate_comprehensive_report()
        
        print(f"\n✅ {args.mode.upper()} mode completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# components/llm_offline.py - Complete offline solution

import os
import random
import re
from typing import Dict, List, Any

class OfflineAIModel:
    """Offline AI model that provides intelligent responses without downloads"""
    
    def __init__(self):
        self.knowledge_base = self._load_biomedical_knowledge()
        self.conversation_history = []
    
    def _load_biomedical_knowledge(self) -> Dict[str, List[str]]:
        """Load built-in biomedical knowledge base"""
        return {
            "neurodegenerative": [
                "Neurodegenerative disorders involve progressive loss of neuronal structure and function.",
                "Key biomarkers for early diagnosis include amyloid-β, tau proteins, and neuroinflammatory markers.",
                "Advanced imaging techniques like PET scans can detect pathological changes before symptom onset.",
                "Stem cell therapy shows promise for replacing damaged neurons in Parkinson's and Alzheimer's disease.",
                "Disease-modifying treatments focus on slowing progression rather than just managing symptoms."
            ],
            "diabetes": [
                "The gut-brain axis plays a crucial role in glucose metabolism and insulin sensitivity.",
                "Gut microbiota composition differs significantly between diabetic and healthy individuals.",
                "Short-chain fatty acids produced by gut bacteria influence insulin signaling pathways.",
                "Probiotics and prebiotics may help improve glycemic control in diabetic patients.",
                "The vagus nerve mediates communication between gut microbiota and pancreatic β-cells."
            ],
            "cancer": [
                "Resistance to targeted therapies often involves activation of alternative signaling pathways.",
                "Tumor heterogeneity contributes to the development of drug resistance mechanisms.",
                "Combination therapies targeting multiple pathways can overcome resistance.",
                "Liquid biopsies enable monitoring of circulating tumor DNA to detect resistance mutations.",
                "Immunotherapy resistance may involve upregulation of immune checkpoint inhibitors."
            ],
            "research_methods": [
                "Systematic reviews and meta-analyses provide the highest level of evidence.",
                "Randomized controlled trials are the gold standard for testing interventions.",
                "Case-control studies are useful for investigating rare diseases or outcomes.",
                "Cohort studies can establish temporal relationships between exposures and outcomes.",
                "Cross-sectional studies provide snapshots of health status in populations."
            ],
            "biomarkers": [
                "Biomarkers must be validated for sensitivity, specificity, and clinical utility.",
                "Proteomic approaches can identify novel protein biomarkers in disease states.",
                "MicroRNAs serve as promising biomarkers for various diseases due to their stability.",
                "Multi-omics integration improves biomarker discovery and validation.",
                "Liquid biopsies offer non-invasive alternatives to tissue-based biomarkers."
            ],
            "therapeutics": [
                "Precision medicine tailors treatments based on individual genetic profiles.",
                "Gene therapy approaches include gene replacement, gene editing, and gene silencing.",
                "Monoclonal antibodies provide targeted therapy with fewer side effects.",
                "Nanotechnology enables targeted drug delivery to specific tissues or cells.",
                "Combination therapies often show synergistic effects in treatment outcomes."
            ]
        }
    
    def _categorize_query(self, query: str) -> str:
        """Categorize the user query based on keywords"""
        query_lower = query.lower()
        
        # Define keyword mappings
        category_keywords = {
            "neurodegenerative": ["alzheimer", "parkinson", "neurodegenerat", "dementia", "brain", "neuron"],
            "diabetes": ["diabetes", "glucose", "insulin", "gut", "microbiota", "microbiome"],
            "cancer": ["cancer", "tumor", "oncolog", "chemotherapy", "resistance", "metastasis"],
            "research_methods": ["study", "research", "method", "trial", "analysis", "systematic"],
            "biomarkers": ["biomarker", "diagnostic", "protein", "rna", "detection"],
            "therapeutics": ["treatment", "therapy", "drug", "medicine", "intervention"]
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or default
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return "research_methods"  # Default category
    
    def _generate_response(self, query: str, category: str) -> str:
        """Generate an intelligent response based on query and category"""
        relevant_facts = self.knowledge_base.get(category, self.knowledge_base["research_methods"])
        
        # Select 2-3 relevant facts
        selected_facts = random.sample(relevant_facts, min(3, len(relevant_facts)))
        
        # Create a coherent response
        response_parts = [
            f"Based on current biomedical research regarding {category.replace('_', ' ')}:",
            ""
        ]
        
        for i, fact in enumerate(selected_facts, 1):
            response_parts.append(f"{i}. {fact}")
        
        response_parts.extend([
            "",
            "For more detailed information, I recommend consulting recent peer-reviewed publications in this field.",
            "",
            "Note: This response is generated from a built-in knowledge base. For current research updates, please consult recent literature."
        ])
        
        return "\n".join(response_parts)
    
    def invoke(self, prompt_text: Any) -> str:
        """Main method to generate responses"""
        try:
            # Handle different input types
            if isinstance(prompt_text, dict):
                if 'text' in prompt_text:
                    query = prompt_text['text']
                elif 'question' in prompt_text:
                    query = prompt_text['question']
                else:
                    query = str(prompt_text)
            else:
                query = str(prompt_text)
            
            # Store in conversation history
            self.conversation_history.append(query)
            
            # Handle greeting and basic queries
            if any(word in query.lower() for word in ['hello', 'hi', 'help', 'what can you do']):
                return self._get_greeting_response()
            
            # Categorize and respond
            category = self._categorize_query(query)
            response = self._generate_response(query, category)
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def _get_greeting_response(self) -> str:
        """Generate greeting response"""
        return """Hello! I'm your offline biomedical research assistant. 

I can help you with questions about:
• Neurodegenerative disorders (Alzheimer's, Parkinson's, etc.)
• Diabetes and metabolic disorders
• Cancer research and therapeutics
• Research methodologies
• Biomarkers and diagnostics
• Therapeutic approaches

While I'm running in offline mode, I can provide evidence-based information from my built-in knowledge base. 

What would you like to know about biomedical research?"""

class NetworkTestModel:
    """Test if we can access any models through alternative methods"""
    
    def __init__(self):
        self.status = self._test_connectivity()
    
    def _test_connectivity(self):
        """Test different ways to access models"""
        tests = {
            "huggingface_hub": False,
            "transformers": False,
            "local_cache": False
        }
        
        try:
            # Test 1: Check if we can import transformers
            import transformers
            tests["transformers"] = True
        except:
            pass
        
        try:
            # Test 2: Check local cache
            import os
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            if os.path.exists(cache_dir):
                tests["local_cache"] = True
        except:
            pass
        
        try:
            # Test 3: Test HF Hub connectivity (with timeout)
            import requests
            response = requests.get("https://huggingface.co", timeout=5)
            if response.status_code == 200:
                tests["huggingface_hub"] = True
        except:
            pass
        
        return tests
    
    def get_status_report(self):
        """Get detailed status report"""
        report = ["🔍 **System Status Report**", ""]
        
        for test, result in self.status.items():
            status = "✅ Available" if result else "❌ Unavailable"
            report.append(f"• {test.replace('_', ' ').title()}: {status}")
        
        report.extend([
            "",
            "**Recommendations:**",
            "• Use offline mode for immediate access",
            "• Contact IT support for network/SSL issues",
            "• Try using a personal network/hotspot",
            "• Consider using Ollama for local models"
        ])
        
        return "\n".join(report)

# Main initialization functions
def initialize_huggingface():
    """Initialize with offline mode"""
    network_test = NetworkTestModel()
    return {
        "logged_in": False, 
        "mode": "offline",
        "status": network_test.get_status_report()
    }

def setup_distilgpt2():
    """Setup offline model"""
    print("🔄 Initializing offline biomedical AI assistant...")
    return OfflineAIModel()

def setup_dialogpt_small():
    """Setup offline model"""
    print("🔄 Initializing offline biomedical AI assistant...")
    return OfflineAIModel()

def setup_tinyllama():
    """Setup offline model"""
    print("🔄 Initializing offline biomedical AI assistant...")
    return OfflineAIModel()

# Test the offline model
if __name__ == "__main__":
    model = OfflineAIModel()
    
    test_queries = [
        "Hello!",
        "How can imaging techniques help diagnose Alzheimer's disease?",
        "What is the role of gut microbiota in diabetes?",
        "How do cancer cells develop resistance to therapy?"
    ]
    
    for query in test_queries:
        print(f"\nQ: {query}")
        print(f"A: {model.invoke(query)}")
        print("-" * 50)